from ctypes import c_char_p, c_int, c_void_p, cdll

import numpy as np
import numpy.ctypeslib as npct
import torch
from torch import Tensor

double_arr = npct.ndpointer(dtype=np.float64, ndim=1, flags="CONTIGUOUS")


class MG5AmplitudeWrapper:
    def __init__(
        self,
        api_path: str,
        param_card: str,
    ):
        self.mg5_lib = cdll.LoadLibrary(api_path)

        self.mg5_lib.process_build.argtypes = None
        self.mg5_lib.process_build.restype = c_void_p
        self.mg5_lib.process_initProc.argtypes = [c_void_p, c_char_p]
        self.mg5_lib.process_initProc.restype = c_void_p
        self.mg5_lib.get_nprocesses.argtypes = [c_void_p]
        self.mg5_lib.get_nprocesses.restype = c_int
        self.mg5_lib.get_me2_vec.argtypes = [
            c_void_p,
            double_arr,
            double_arr,
            c_int,
            c_int,
        ]
        self.mg5_lib.get_me2_vec.restype = None
        self.mg5_lib.process_free.argtypes = [c_void_p]
        self.mg5_lib.process_free.restype = None

        self.process = self.mg5_lib.process_build()
        self.nsub_processes = self.mg5_lib.get_nprocesses(self.process)

        # Set and load parameters
        self.mg5_lib.process_initProc(self.process, param_card.encode("ascii"))

    def me2(self, p: Tensor) -> Tensor:
        nbatch = p.shape[0]
        nparticles = p.shape[1]
        p_flat = p.reshape((nbatch, 4 * nparticles)).flatten().numpy()
        m2_out = np.empty((nbatch, self.nsub_processes)).flatten()
        self.mg5_lib.get_me2_vec(
            self.process,
            p_flat,
            m2_out,
            nbatch,
            nparticles,
        )
        m2_all = m2_out.reshape((nbatch, self.nsub_processes))
        return torch.from_numpy(m2_all[:, 0])

    # def __del__(self):
    #     self.mg5_lib.process_free(self.process)
