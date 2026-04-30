from abc import ABC, abstractmethod
from collections import defaultdict

# Intel packages
from nxcore.arch.n3b.n3board import N3Board
import nxkernel as nxk


class BaseNxModel(ABC):

    @abstractmethod
    def build_model(self):
        """build_model must set self.model and call self.model.setup()"""
        pass

    @abstractmethod
    def _make_addrs_list(self, model: nxk.Module):
        """_make_addrs_list must partition self.model and a address list for the loihi neurocore"""
        pass

    def forward(self, num_steps: int, nxstate_map: dict = dict()) -> dict:
        """Run model for num_steps and retrieve nxstates.

        Args:
            num_steps (int): number of timesteps
            nxstate_map (dict, optional): dict of nxstates to retrieve
                Keys are group name and value is list of nxstate names. Defaults to dict().

        Returns:
            dict: dict of nxstates
        """

        addrs_list = self._make_addrs_list()

        board = N3Board()
        self.model.to_nxcore(board, addrs_list)

        nxstates = defaultdict(dict)
        try:
            board.run(num_steps)
        except Exception as e:
            print(e)
        finally:
            board.fetchAll()

            for group_name, nxstate_list in nxstate_map.items():
                for nxstate_name in nxstate_list:
                    group = getattr(self.model, group_name)
                    nxstate = getattr(group.neuron, nxstate_name).get(board)
                    nxstates[group_name][nxstate_name] = nxstate

            board.stop()

        return nxstates
