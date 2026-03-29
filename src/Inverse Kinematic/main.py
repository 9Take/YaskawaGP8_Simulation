import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

Q_MIN = np.array([-2.97, -1.75, -3.14, -3.49, -2.09, -6.28])
Q_MAX = np.array([ 2.97,  2.62,  1.22,  3.49,  2.09,  6.28])
GP8_MAX_REACH = 0.727


