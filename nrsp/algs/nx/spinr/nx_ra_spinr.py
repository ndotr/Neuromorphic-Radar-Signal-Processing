# Public packages
import numpy as np
from collections import OrderedDict
import pathlib
import os
import logging
import time
import platform
import datetime
import pandas as pd


# Intel packages
import nxkernel as nxk
import nxkernel.kernels as K
import nxcore.arch.n3b.n3board
from nxkernel.kernels.utils.weight_utils import SignMode
from nxkernel.utils import characterization as ch
import nrsp.algs.nx.spinr.nx_spinr

#nxk.Logger.set_nxcore_log_level(logging.DEBUG)

# Custom packages
import nrsp.utils.nx

class NxRASpiNRModel():
    """
    Range-Angle-SpiNR Model.

    Architecture:

    Attributes:
        n_channels (int):
        n_neurons_0 (int):
        weights (np.array):
        omega_range (float):
        alpha_mag_smooth (float):
        log (logging.log):
        TODO:
        

    Methods:
        TODO:

    """
    def __init__(self, n_channels, n_angles, n_ranges,
                 alpha_grad, thresh, thresh_silent, start_time,
                 turnoff_neurons=0,
                 log=None):
        """
        TODO:

        """
        self.log = log
        self.log.debug("Initialize NxRASpiNRModel() ...")
        self.turnoff_neurons = turnoff_neurons
        self.n_timesteps = None

        self.n_channels = n_channels
        self.n_angles = n_angles
        self.n_ranges = n_ranges
        self.n_neurons = n_ranges * n_angles
        
        self.omega = self._phasor_weights(n_ranges, perc=0.5)
        self.omega = np.repeat(self.omega, self.n_angles)
        self.log.debug("Shape of omega: {}".format(self.omega.shape))
        self.weights = self._steering_weights(in_dims=n_channels, out_dims=n_angles).T
        self.weights = np.tile(self.weights, self.n_ranges).T*127
        self.log.debug("Shape of weights: {}".format(self.weights.shape))

        #self.spinr_ucode_path = pathlib.Path('nrsp/algs/nx/spinr/kernels/spinr_word_rspike.dasm').resolve()
        #self.spinr_ucode_args = {
        #                        'alpha_grd': (alpha_grad*2**15),
        #                        'beta_grd': ((1 - alpha_grad)*2**15),
        #                        'grd_th': u_threshold,
        #                        } 

        #self.spinr_ucode_path = pathlib.Path('nrsp/algs/nx/spinr/kernels/spinr_word_rspike_slim.dasm').resolve()
        self.spinr_ucode_path = pathlib.Path('nrsp/algs/nx/spinr/kernels/spinr_long_rspike_turnoff.dasm').resolve()
        #self.spinr_ucode_path = pathlib.Path('nrsp/algs/nx/spinr/kernels/spinr_long_rspike_dummy.dasm').resolve()
        self.spinr_ucode_args = {
                                'alpha_grd': (alpha_grad*2**15),
                                'beta_grd': ((1 - alpha_grad)*2**15),
                                'thresh_silent': thresh_silent,
                                'thresh': thresh,
                                'start_time': start_time,
                                } 
        
        self.model = nxk.Module()
        self.mesh = nxk.system.mesh.OheoGulchMesh()
        self.viz = ch.visualize.WorkloadVisualizer(self.mesh)
        self.stats = {}
        self.log.debug("Initialized NxRASpiNRModel().")

    def build_cyclic_buffer_model(self, input_data):

        self.n_timesteps = input_data.shape[1]
        input_groups = self._build_cyclic_buffers(input_data=input_data)
        self._add_groups(input_groups)
        spinr_group = self._build_spinr_neurons(weights=self.weights, omega=self.omega, 
                                                ucode_args=self.spinr_ucode_args,
                                                ucode_path=self.spinr_ucode_path)
        self._add_groups(spinr_group)

        self.model.setup()

    def build_ethernet_model(self):

        input_groups = self._build_ethernet_input()
        self._add_groups(input_groups)
        spinr_group = self._build_spinr_neurons(weights=self.weights, omega=self.omega, 
                                                ucode_args=self.spinr_ucode_args,
                                                ucode_path=self.spinr_ucode_path)
        self._add_groups(spinr_group)
        output_group = self._build_ethernet_output()
        self._add_groups(output_group)

        self.model.setup()

    def partition_cyclic_buffer_model(self):

        #tile_shapes=(self.n_channels, self.n_ranges//4)
        #tile_shapes=(self.n_channels//1, self.n_neurons//(116))
        tile_shapes=(self.n_channels//2,  32)

        self.model.partition(tile_shapes=tile_shapes)
        for r in self.model.resources: print(r, end='\n\n')

        addrs_list = [
                [nxk.make_addr(chip=0, core_idx=idx) for idx in range(2)],
                [
                 nxk.make_addr(chip=0, core_idx=idx+2) for idx in range(self.model.groups[1].num_cores)
                 ]
            ]  
        #addrs_list = []
        #nc_offset = 0
        #for group in self.model.groups:
        #    addrs_list.append([nxk.make_addr(chip=0, core_idx=idx + nc_offset) for idx in range(group.num_cores)])
        #    nc_offset += group.num_cores
        
        return tile_shapes, addrs_list

    def partition_ethernet_model(self):

        ETH_INTERFACE = os.environ.get('ETH_INTERFACE', 'eth0')
        ETH_MAC = os.environ.get('ETH_MAC', '0x3c6d6602b60f')
        LOIHI_MAC = os.environ.get('LOIHI_MAC', '0x0015edbeefed')

        #tile_shapes=(self.n_channels, self.n_neurons/119, self.n_neurons)
        #tile_shapes=(self.n_channels//1, self.n_neurons//112, self.n_neurons)
        tile_shapes=(self.n_channels//1, 1344, self.n_neurons)

        self.model.partition(tile_shapes=tile_shapes)

        eth_chip_idx = 0

        eth_chip_loc = self.mesh.get_chip_loc(idx=eth_chip_idx)
        ethernet_addr = nxk.make_addr(chip_idx=eth_chip_idx,
                                    chip_loc=eth_chip_loc,
                                    ethernet_interface=ETH_INTERFACE,
                                    ethernet_mac_address=ETH_MAC,
                                    loihi_mac_address=LOIHI_MAC)

        addrs_list = [[ethernet_addr],
                    [nxk.make_addr(chip_idx=eth_chip_idx, core_idx=idx)
                    for idx in range(self.model.spinr.num_cores)],
                    [ethernet_addr]]
        self.log.debug("Created addrs_list: ")
        self.log.debug(addrs_list)

        return tile_shapes, addrs_list

    def init_board(self, addrs_list):

        self.board = nxcore.arch.n3b.n3board.N3Board()
        self.model.to_nxcore(self.board, addrs_list) 

    def save_model_image(self, filename):

        self.model.connectivity('LR').save(filename)



    def run_cyclic_buffer_stats(self, n_timesteps=None):
        
        if n_timesteps is None:
            n_timesteps = self.n_timesteps

        self._init_stats(n_timesteps)

        cxstate_struct = self.model.spinr.neuron.cxstate_struct
        states = {}
        for key in cxstate_struct.keys(): #self.model.rf.neuron.cxstate_struct.keys():
            states[key] = []

        try:
            self.log.debug("Try running for {} timesteps ...".format(self.n_timesteps))
            self.board.run(n_timesteps)
        finally:
            self.log.debug("Finished running.")
            #self.stats["power"].collect_stats(self.model, self.board)
            self.stats["activity"].collect_stats(self.model, self.board)
            #self.stats["memory"].collect_stats(self.model, self.board)
            self.stats["traffic"].collect_stats(self.model, self.board)
            self.stats["mac"].collect_stats(self.model)
            self.stats["sparsity"].collect_stats(self.model, self.board)

            for key in cxstate_struct.keys():
                states[key].append(self.model.spinr.neuron.get(board=self.board, field=cxstate_struct[key]))
            for key in cxstate_struct.keys():
                states[key] = np.array(states[key])

            self.board.fetchAll()
            self.board.stop()
            print("Finished stopping.")

        return states, self.stats

    def run_cyclic_buffer(self, n_timesteps=1):

        cxstate_struct = self.model.spinr.neuron.cxstate_struct
        states = {}
        for key in cxstate_struct.keys(): #self.model.rf.neuron.cxstate_struct.keys():
            states[key] = []

        self.stats["py_runtime"] = np.zeros((self.n_timesteps, 3))

        try:
            self.log.info("Try running for {} timesteps ...".format(self.n_timesteps//n_timesteps))
            for t in range(self.n_timesteps//n_timesteps):
                self.stats["py_runtime"][t,0] = time.time()
                self.board.run(n_timesteps)
                self.stats["py_runtime"][t,2] = time.time()
                for key in cxstate_struct.keys():
                    states[key].append(self.model.spinr.neuron.get(board=self.board, field=cxstate_struct[key]))
        finally:
            for key in cxstate_struct.keys():
                states[key] = np.array(states[key])
            print("Finished running.")
            self.board.fetchAll()
            self.board.stop()
            print("Finished stopping.")

        return states
    
    def run_ethernet(self, input_data):

        n_timesteps = input_data.shape[0]
        self.log.debug("Input shape: {}".format(input_data.shape))
        output = np.zeros((n_timesteps, self.n_neurons))

        try:
            self.log.debug("Try running for {} timesteps ...".format(n_timesteps))
            self.model.ethernet_init(self.board)
            self.board.run(n_timesteps, aSync=True)
            self.model.start()
            for t in range(n_timesteps):
                self.log.debug("Sending at time: {} ...".format(t))
                self.model.input_real.send(input_data.real[t])
                self.log.debug("Sent.")
                self.log.debug("Receiving at time: {} ...".format(t))
                output[t] = self.model.output.recv()
                self.log.debug("Received.")
        finally:
            self.model.stop()
            self.board.stop()

        return output

    def run_ethernet_stats(self, input_data):

        n_timesteps = input_data.shape[0]
        self.log.debug("Input shape: {}".format(input_data.shape))
        output = np.zeros((n_timesteps, self.n_neurons))

        self._init_stats(n_timesteps)
        self.stats["py_runtime"] = np.zeros((n_timesteps, 3))

        try:
            self.log.debug("Try running for {} timesteps ...".format(n_timesteps))
            self.model.ethernet_init(self.board)
            self.board.run(n_timesteps, aSync=True)
            self.model.start()
            comp_time = time.time()
            for t in range(n_timesteps):
                self.stats["py_runtime"][t,0] = time.time()
                self.model.input_real.send(input_data.real[t])
                self.stats["py_runtime"][t,1] = time.time()
                output[t] = self.model.output.recv()
                self.stats["py_runtime"][t,2] = time.time()
        finally:
            #self.stats["power"].collect_stats(self.model, self.board)
            self.stats["activity"].collect_stats(self.model, self.board)
            self.stats["memory"].collect_stats(self.model, self.board)
            self.stats["traffic"].collect_stats(self.model, self.board)
            self.stats["mac"].collect_stats(self.model)
            self.stats["sparsity"].collect_stats(self.model, self.board)
            self.model.stop()
            self.board.stop()
            #self.board.disconnect()
            self.stats["py_runtime"] -= comp_time

        return np.array(output), self.stats

    def build_report(self, filename):

        exec_model = ch.model.LinearExecutionModel()
        exec_model.estimate(self.stats["activity"], self.stats["traffic"])

        stats = dict(date=str(datetime.datetime.now()),
                     node=platform.node())
        if 'PARTITION' in os.environ.keys():
            stats['PARTITION'] = os.environ['PARTITION']
        if 'LOIHI_GEN' in os.environ.keys():
            stats['LOIHI_GEN'] = os.environ['LOIHI_GEN']
        if 'BOARD' in os.environ.keys():
            stats['BOARD'] = os.environ['BOARD']
        if 'NXSDKHOST' in os.environ.keys():
            stats['NXSDKHOST'] = os.environ['NXSDKHOST']
        if 'NXOPTIONS' in os.environ.keys():
            stats['NXOPTIONS'] = os.environ['NXOPTIONS']
        metadata = pd.DataFrame(data=stats.values(), index=stats.keys(), columns=[''])

        reporter = ch.report.Reporter(
            model=self.model,
            report_template='./report_template.md',
            visualizer=self.viz,
            #power_stats=self.stats["power"].stats,
            #runtime_stats=self.stats["runtime"].stats,
            stats=[self.stats["mac"],
                   self.stats["activity"],
                   exec_model,
            #       #self.stats["power"],
                   self.stats["sparsity"],
                   self.stats["traffic"],
                   #self.stats["memory"]
                   ],
            summary=pd.DataFrame(data=[self.n_timesteps],
                                 index=['num_steps'],
                                 columns=['Test Summary']),
            metadata=metadata,
        )

        reporter.notes='Here you can express the notes on the experiment results'
        reporter.report(path=filename, overwrite=True)

    def _add_groups(self, groups):
        
        if  type(groups) is list: 
            for group in groups:
                self.model.add_group(group)
        else:
            self.model.add_group(groups)

    def _build_cyclic_buffers(self, input_data):

        self.log.debug("Building CyclicBuffer() ...")
        self.log.debug("Number of timesteps: {}".format(input_data.shape[-1]))
        self.log.debug("Real input shape: {}".format(input_data.real.shape))
        self.log.debug("Imag input shape: {}".format(input_data.imag.shape))

        self.n_timesteps = input_data.shape[-1]
        input_real_group = nxk.NcGroup(nxk.kernels.CyclicBuffer(input_data.real, num_msg_bytes=2), name='input_real')
        #input_imag_group = nxk.NcGroup(nxk.kernels.CyclicBuffer(input_data.imag, num_msg_bytes=2), name='input_imag')

        self.log.debug("Done Building CyclicBuffer().")

        return input_real_group
    
    def _build_ethernet_input(self):

        input_real_group = nxk.EthernetInputGroup((self.n_channels, ), name='input_real')
        #input_imag_group = nxk.EthernetInputGroup((self.n_channels, ), name='input_imag')

        return input_real_group

    def _build_ethernet_output(self):

        output_group = nxk.EthernetOutputGroup((self.n_neurons, ), 
                                                in_ports=[self.model.spinr.id - self.model.next_id],
                                               name='output')

        return output_group


    def _build_spinr_neurons(self, weights, omega, ucode_args, ucode_path):

        # Define synapses
        #syn_args = dict(sparse_packing=False, use_shared_axon=True, optimize_weights=False, num_weight_bits=7,
        #                weight_exp=0)
        #syn_args = dict(sparse_packing=False, use_shared_axon=True)
        syn_args = dict(sparse_packing=True, use_shared_axon=False, optimize_weights=False, num_weight_bits=8)
        synapses = [K.Linear(weight=weights.real, **syn_args, name='real_wgt'), 
                    K.Linear(weight=weights.real, **syn_args, name='imag_wgt')]
        #synapses = [
        #            K.Linear(weight=K.interleave([self.weights[...,0], self.weights[...,1]]), name='real_input_wgt'),
        #            K.Linear(weight=K.interleave([-self.weights[...,1], self.weights[...,0]]), name='imag_input_wgt'),
        #            ]

        # Define neuron
        neuron_args = dict(
            ucode_path=ucode_path,
            ucode_args=ucode_args,
            lct=(omega.real * 2**15).astype(np.int32),
            lst=(omega.imag * 2**15).astype(np.int32),
            turnoff_neuron=int(self.turnoff_neurons),
        )
        
        #init_cxstates = {} 
        neuron = K.Neuron(shape=(self.n_neurons,), 
                            out_spike_type='ls', 
                            **neuron_args)
                           # **init_cxstates)
       
        spinr_group = nxk.NcGroup(neuron=neuron,
                                synapses=synapses,
                                #interleaved_da=True,
                                in_ports=[self.model.input_real.id - self.model.next_id,
                                          self.model.input_real.id - self.model.next_id],
                                da_ports=[0,1],
                                name='spinr')

        return spinr_group


    def _init_stats(self, n_timesteps):


        #self.stats["power"] = ch.stats.PowerStats(self.board, n_timesteps, self.mesh)
        self.stats["runtime"]       = ch.stats.RuntimeStats(self.board, t_start=1, t_end=n_timesteps-1)
        self.stats["py_runtime"]    = None
        self.stats["activity"]      = ch.stats.ActivityStats(num_steps=n_timesteps)
        self.stats["memory"]        = ch.stats.MemoryStats()
        self.stats["traffic"]       = ch.stats.TrafficStats()
        self.stats["mac"]           = ch.stats.MACStats()
        self.stats["sparsity"]      = ch.stats.SparsityStats(activity_stats=self.stats["activity"], mac_stats=self.stats["mac"])
       


    
    def _phasor_weights(self, n_bins, perc=1):

        return np.exp(1j*np.linspace(0,perc-perc/n_bins,n_bins)*np.pi*2)    
    
    def _steering_weights(self, in_dims, out_dims):

        W = np.zeros((out_dims, in_dims)).astype('complex128')
        for o in range(out_dims):
            for i in range(in_dims):
                phi = 2*np.pi*i*(o-out_dims//2)/out_dims
                #if phi == 0:
                #    W[a,rx] = 0
                #else:
                W[o,i] = np.exp(1j*phi)

        return W