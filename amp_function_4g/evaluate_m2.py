import os

import torch
import yaml
from madspace.cuts import PhaseSpaceCuts
from madspace.rambo import RamboOnDiet

from me_wrapper import MG5AmplitudeWrapper
from utils import Sigma_ij

import matplotlib.pyplot as plt
import numpy as np

# IMPORTANT!
torch.set_default_dtype(torch.float64)

##=============== Definitions ==================##

# Some definitions
# Particle PIDS
GLUON = 21

# Final state pids requires
# With this, the cutter knows on which particles to cut
PIDS = [GLUON, GLUON, GLUON, GLUON]  # PIDs of outgoing particles
NPARTICLES = 4  # number of outgoing particles
E_BEAM = 1000  # should be the beam energy from your madgraph run_card

##=============== GENERATE THE MOMENTA ==================##

# Initialize phase-space generator
# Rambo samples flat, so weight = const
rambo = RamboOnDiet(NPARTICLES)

# sample momenta
n = int(1e4)
r = torch.rand((n, 3 * NPARTICLES - 4))
energy = E_BEAM * torch.ones(n)
(p_ext,), weight = rambo.map([r, energy])

# Output of momenta is shape (n, NPARTICLES + 2, 4)
# including initial state

##=============== EVALUATE ME2 ==================##

dirname = os.path.dirname(__file__)
API_PATH = os.path.join(dirname, "cpp/SubProcesses/P1_Sigma_sm_gg_gggg/mg5_vectorized.so")
PARAM_CARD = os.path.join(dirname, "Cards/param_card.dat")
mg5_api = MG5AmplitudeWrapper(API_PATH, PARAM_CARD)
m2_real = mg5_api.me2(p_ext)

print(m2_real.shape)
print(m2_real[:5])
