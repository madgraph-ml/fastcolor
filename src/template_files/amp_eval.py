import numpy as np
from typing import Callable, Optional
import numpy.ctypeslib as npct
from ctypes import c_int, cdll
import torch

double_arr = npct.ndpointer(dtype=np.float64, ndim=1, flags="CONTIGUOUS")
int_arr = npct.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")

API_PATH = "lib_matrix.so"
mg_lib = cdll.LoadLibrary(API_PATH)
mg_lib.get_batched_matrix.argtypes = [
    double_arr,
    c_int,
    c_int,
    double_arr,
]
mg_lib.get_batched_matrix.restype = None
        

# TODO: get momenta
mom_data = torch.load('PATH_TO_MOMENTA')
momenta = mom_data # do sth with mom_data to get the momenta?
n_events = mom_data # number of events
n_part = # number of particles of the process (both intial and final)

# flatten it
mom_flat = momenta.flatten()

# flat matrix output
# this will be filled
matrix_out = np.empty((n_events,))

# get momenta in batches way
mg_lib.get_batched_matrix(mom_flat, n_events, n_part, matrix_out)

# TODO: Save momenta with matrix elements into file

print(matrix_out)