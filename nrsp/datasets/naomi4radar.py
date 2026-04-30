import numpy as np
import scipy.signal
from scipy import signal
from scipy import fft
from scipy import io as sio
import os


def raw2az_el_vx(raw, n_tx, n_az_channels, n_el_channels):

    time_data = raw2time(raw, n_tx)
    vx_data = time2vx(time_data)
    az_el_vx_data = vx2az_el_vx(vx_data, n_az_channels=n_az_channels, n_el_channels=n_el_channels)

    return az_el_vx_data

def raw2time(raw, n_tx):
    """
    Input:  (n_samples, n_rx, n_chirps)
    Output: (n_rx, n_tx, n_samples, n_chirps_per_tx)
    """
    n_samples, n_rx, n_chirps = raw.shape
    n_chirps_per_tx = n_chirps // n_tx
    time_data = raw.reshape((n_samples, n_rx, n_tx, n_chirps_per_tx), order='F')
    return np.transpose(time_data, (1, 2, 0, 3))

def time2vx(time_data):
    """
    Input:  (n_rx, n_tx, n_samples, n_chirps_per_tx)
    Output: (n_vx, n_samples, n_chirps_per_tx)
    """

    n_rx, n_tx, n_samples, n_chirps_per_tx = time_data.shape
    n_vx = n_rx * n_tx
    # ---- MIMO Re-arrangement ----
    # [Rx, Tx, samples, chirps]
    vx_data = np.zeros((n_vx, n_samples, n_chirps_per_tx), dtype=time_data.dtype)
    for txi in range(16):
        for blk in range(4):
            src_idx = slice(blk * 4, blk * 4 + 4)
            dst_idx = slice(blk * 64 + txi * 4, blk * 64 + txi * 4 + 4)
            vx_data[dst_idx, :, :] = time_data[src_idx, txi, :, :].squeeze()
    return vx_data

def vx2az_el_vx(vx_data, n_az_channels, n_el_channels):
    """
    Input:  (n_vx, n_samples, n_chirps_per_tx)
    Output: (n_az_channels, n_el_channels, n_samples, n_chirps_per_tx)
    """

    n_vx, n_samples, n_chirps_per_tx = vx_data.shape
    Vx_Tx_idx = np.array([24, 0, 52, 44, 28, 12, 56, 40, 20, 4, 48, 32, 16, 8, 60, 36])
    Vx_Rx_idx = np.array([1, 2, 67, 68, 65, 66, 3, 4, 193, 194, 131, 132, 129, 130, 195, 196]) - 1

    # [Az channels, El channels, n_samples, n_chirps_per_tx]
    az_el_vx_data = np.zeros((n_az_channels, n_el_channels, n_samples, n_chirps_per_tx), dtype=vx_data.dtype)

    for tx_group in range(4):
        for tx_idx_in_group in range(4):
            full_tx_idx = tx_group * 4 + tx_idx_in_group
            start_idx = tx_group * 16
            target_indices = start_idx + np.arange(16)
            src_indices = Vx_Tx_idx[full_tx_idx] + Vx_Rx_idx
            az_el_vx_data[target_indices, tx_idx_in_group, :, :] = vx_data[src_indices, :, :] 

    return az_el_vx_data

def az_el_vx2non_uniform_vx(az_el_vx_data, n_padded_az):

    n_az_channels, n_el_channels, n_range, n_chirps_per_tx = az_el_vx_data.shape

    lambda0=1
    d_tx_az = np.array([0, 4.5, 16, 20.5]) * lambda0
    d_rx_az = np.arange(16) * lambda0
    d_vir_az = np.array([tx + rx for tx in d_tx_az for rx in d_rx_az])
    FFT_Vx_idx = np.argsort(d_vir_az[:64])

    padded_az_el_range_data = np.zeros((n_padded_az, n_el_channels, n_range, n_chirps_per_tx), dtype=az_el_vx_data.dtype)
    valid_indices = [0, 2, 4, 6] + list(range(8, 64)) + [65, 67, 69, 71]
    padded_az_el_range_data[valid_indices, :, :, :] = az_el_vx_data[FFT_Vx_idx, :, :, :]

    return padded_az_el_range_data

def generate_calib_vec(range_data, axis, idx=None, filename=False):

    axes = tuple(i for i in range(range_data.ndim) if i != axis)
    if idx is None:
        idx = np.argmax(np.mean(np.abs(range_data), axis=axes))

    calib_vec = np.mean(np.take(range_data, idx, axis=axis), axis=-1)
    calib_angle = np.angle(calib_vec)
    calib_mag = np.abs(calib_vec)
    calib_mag = np.max(calib_mag) / calib_mag
    calib_vec = calib_mag * np.exp(-1j * calib_angle)
    if filename:
        np.save("out/calib_vec.npy", calib_vec)

    return calib_vec

def broadcast_1d(vec, data, axis):

    shape = [1] * data.ndim
    shape[axis] = -1

    return vec.reshape(shape)

def apply_broadcast_1d(vec, data, axis, normalize=False):

    shape = [1] * data.ndim
    shape[axis] = -1
    ret = data * vec.reshape(shape)
    if normalize:
        ret *= 1.0 / np.sum(vec)

    return ret

def db(x, ref=1.0, floor_db=-100):
    """Convert linear magnitude to dB scale."""
    with np.errstate(divide='ignore'):
        db_val = 20 * np.log10(np.maximum(x / ref, 1e-12))  # Avoid log(0)
    return np.maximum(db_val, floor_db)

def broadcast_1d_chebwin(data, axis, at):

    shape = [1] * data.ndim
    n_samples = data.shape[axis]
    shape[axis] = -1
    win = scipy.signal.windows.chebwin(n_samples, at=at)
    # Broadcasting

    return win.reshape(shape)

def apply_1d_chebwin(data, axis, at):

    shape = [1] * data.ndim
    n_samples = data.shape[axis]
    shape[axis] = -1
    win = scipy.signal.windows.chebwin(n_samples, at=at)
    # Broadcasting

    return data * win.reshape(shape) * (1.0 / np.sum(win))




def readRawBinCasc(datadir, frameNr, nSamples, nRamps, nChannelsPerCtrx, nCtrx, nFrames):
    
    Nbins = nSamples*nRamps*4
    frameSize = Nbins*2

    fid = 0
    fname = datadir + f"/ctrx{fid}_bin.raw"

    f = open(fname, "rb")
    f.seek(frameSize*frameNr)
    data = f.read(frameSize)
    f.close()
    Raw = np.frombuffer(data, np.uint16)
    Raw = np.reshape(Raw,(4, nSamples, nRamps),'F')

    for fid in range(1,nCtrx):
        fname = datadir + f"/ctrx{fid}_bin.raw"
        f = open(fname, "rb")
        f.seek(frameSize*frameNr)
        data = f.read(frameSize)
        f.close()
        tmp = np.frombuffer(data, np.uint16)
        tmp = np.reshape(tmp,(4, nSamples, nRamps),'F')
        Raw = np.concatenate((Raw,tmp), 0)

    Raw = Raw.transpose((1, 0, 2))

    # remove padded bits:
    Raw = np.bitwise_and(Raw, np.uint16(0xFFF0))
    Raw = Raw.astype(np.int16, copy=False)

    Raw = np.float64(Raw)

    return Raw




###############################################################################
# Backup
###############################################################################
def readRawMat(fname):
    mat_contents = sio.loadmat(fname)
    Raw = mat_contents['ADC_R']
    Raw = np.float64(Raw)
    Raw = np.transpose(Raw, (0, 2, 1))
    return Raw

###############################################################################
def readRawBin(datadir, fid, Ns1, Nrx, Ns2):
    fname = datadir + f"/timedata_{fid:04d}.bin"
    Raw = np.fromfile(fname, np.int16)
    Raw = Raw.reshape((Ns1, Nrx, Ns2))
    Raw = np.float64(Raw)

    return Raw

###############################################################################
def rdFft(Raw):
    Ns1, Nrx, Ns2 = Raw.shape

    Wd1 = signal.windows.chebwin(Ns1, 80)
    Wd2 = signal.windows.chebwin(Ns2, 80)

    Nf1 = 2*Ns1
    Nf2 = 2*Ns2

    Nrang = int(Nf1/2)
    Ndopp = Nf2

    # Range FFT
    Ff1 = np.complex64(np.zeros((Nrang,Nrx,Ndopp)))
    for chirp in range(0, Ns2):
        for rx in range(0, Nrx):
            Rw = Raw[:,rx,chirp] * Wd1
            tmp = fft.fft(Rw, Nf1)
            Ff1[:,rx,chirp] = tmp[0:Nrang]

    # Doppler FFT
    RD = np.complex64(np.zeros((Ndopp,Nrang,Nrx)))
    for rx in range(0, Nrx):
        for rg in range(0, Nrang):
            Rw2 = Ff1[rg,rx,0:Ns2] * Wd2
            #tmp = fft.fft(Rw, Nf2)
            tmp = fft.fftshift(fft.fft(Rw2, Nf2))
            RD[:,rg,rx] = tmp

    return RD

###############################################################################
def nci(RD):
    Ndopp,Nrang,Nrx = RD.shape
    # NCI
    Plin = np.abs(RD)**2
    NCI = Plin[:,:,0]
    for rx in range(1, Nrx):
        NCI = NCI + Plin[:,:,rx]

    return NCI

###############################################################################
def localMax(NCI):
    Ndopp, Nrang = NCI.shape
    LMAP = np.zeros((Ndopp, Nrang), bool)

    NCIt = np.concatenate((NCI[(Ndopp-1):Ndopp,:],NCI,NCI[0:1,:]))
    NCIt = np.concatenate((NCIt[:,1:2],NCIt,NCIt[:,(Nrang-2):(Nrang-1)]),axis=1)

    for r in range(0,Nrang):
        for d in range(0,Ndopp):
            cut = NCIt[d+1,r+1]
            dmax = (cut > NCIt[d+2,r+1]) & (cut > NCIt[d+0,r+1])
            rmax = (cut > NCIt[d+1,r+2]) & (cut > NCIt[d+1,r+0])
            LMAP[d,r] = dmax & rmax

    return LMAP

###############################################################################
def thresholding(NCI, beta_dB):
    Ndopp, Nrang = NCI.shape

    S = np.zeros((Ndopp,Nrang))
    for r in range(0,Nrang):
        tmp = np.sort(NCI[:,r])
        S[:,r] = tmp[int(Ndopp/2)]

    threshold = S * (10**(beta_dB/10))
    TMAP = (NCI > threshold)

    return TMAP

###############################################################################
def matching(TMAP, txCode):
    Ndopp, Nrange = TMAP.shape
    shift = np.int32(txCode*Ndopp)

    Ntx = txCode.size
    MMAP = np.ones((Ndopp,Nrange), bool)
    for tx in range(0, Ntx):
        MMAP = MMAP & np.roll(TMAP, -shift[tx], axis=0)

    return MMAP

###############################################################################
def getPeaks(DMAP):
    Ndopp, Nrange = DMAP.shape
    PEAKS = []
    
    for r in range(1, Nrange):
        for d in range (0, Ndopp):
            if DMAP[d,r]:
                tmp = (r,d)
                PEAKS.append(tmp)
            
    return PEAKS

###############################################################################
def mimoVector(RD, rdIdx, txCode, vIdx):
    #1: collect according to idx and tx code
    #2: arrange according to vIdx
    
    return mVec
    
###############################################################################
def axis(NCI):
    Ndopp, Nrang = NCI.shape

    y = np.linspace(0, Ndopp-1, Ndopp)
    x = np.linspace(0, Nrang-1, Nrang)
    x,y = np.meshgrid(x,y)

    return x,y

###############################################################################
def axisPlotly(NCI):
    Ndopp, Nrang = NCI.shape

    y = np.linspace(0, Ndopp-1, Ndopp)
    x = np.linspace(0, Nrang-1, Nrang)
    x,y = np.meshgrid(x,y)

    return x,y
