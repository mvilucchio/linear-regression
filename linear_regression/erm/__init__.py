from numpy import tanh, ones_like
from numpy import float32 as np_float32
from numba import njit

MAX_ITER_MINIMIZE = 50_000
GTOL_MINIMIZE = 1e-5
XTOL_MINIMIZE = 1e-5

BLEND_GAMP = 1.0
TOL_GAMP = 5e-3
MAX_ITER_GAMP = 5000

MAX_ITER_PDG = 100
STEP_BLOCK_PDG = 10
TOL_PDG = 1e-6
STEP_SIZE_PDG = 1e-2
TEST_ITERS_PDG = 100
N_ALTERNATING_PROJ = 25
NOISE_SCALE = np_float32(1.0)
