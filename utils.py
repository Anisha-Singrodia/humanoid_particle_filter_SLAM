# Pratik Chaudhari (pratikac@seas.upenn.edu)

import numpy as np

cos, sin = np.cos, np.sin

# some lambda functions for typical calculations
get_so2 = lambda a: np.array([[cos(a), -sin(a)],
                            [sin(a), cos(a)]])

get_se2 = lambda a, v: np.array([[cos(a), -sin(a), v[0]],
                     [sin(a), cos(a), v[1]],
                     [0,0,1]])

def euler_to_so3(r,p,y):
    rx = np.array([[1,0,0], [0, cos(r), -sin(r)], [0, sin(r), cos(r)]])
    ry = np.array([[cos(p),0,sin(p)], [0, 1, 0], [-sin(p), 0, cos(p)]])
    rz = np.array([[cos(y),-sin(y),0], [sin(y), cos(y),0],[0, 0, 1]])
    so3 = rz @ ry @ rx
    return so3

def euler_to_se3(r, p, y, v):
    so3 = euler_to_so3(r,p,y)
    se3 = np.vstack((np.hstack((so3, v.reshape(-1,1))), np.array([0,0,0,1])))
    return se3

make_homogeneous_coords_2d = lambda xy: np.vstack((xy, np.ones(xy.shape[1])))
make_homogeneous_coords_3d = lambda xyz: np.vstack((xyz, np.ones(xyz.shape[1])))

def smart_plus_2d(p1,p2):
    """
    See guidance.pdf
    p1, p2 are two poses (x1, y1, yaw1) and (x2, y2, yaw2)
    """
    R = get_so2(p1[2])
    t = p1[:2] + (R @ p2[:2])
    return np.array([t[0], t[1], p1[2]+p2[2]])

def smart_minus_2d(p2, p1):
    """
    See guidance.pdf
    p2, p1 (note the order) are two poses (x2, y2, yaw2) and (x1, y1, yaw1)
    """
    R = get_so2(p1[2])
    t = R.T @ (p2[:2]-p1[:2])
    return np.array([t[0], t[1], p2[2]-p1[2]])



# [[-4.64670052e-10]
#  [ 4.05781452e-04]
#  [ 0.00000000e+00]]
#   0%|                                                                                                                     | 0/10 [00:00<?, ?it/s]ye [4.96709506e-05 3.91955022e-04 6.47688538e-05]
# ye [2.01973936e-04 3.68539684e-04 4.13551581e-05]
# ye [ 3.59895218e-04  4.45283157e-04 -5.59228048e-06]
# ye [ 4.14151222e-04  3.98941388e-04 -5.21652558e-05]
# ye [ 0.00043835  0.00020761 -0.00022466]
# ye [ 0.00038212  0.00010633 -0.00019323]
# ye [ 2.91316289e-04 -3.49001189e-05 -4.66674289e-05]
# ye [ 2.68738659e-04 -2.81472984e-05 -1.89142248e-04]
# ye [ 2.14300386e-04 -1.70550395e-05 -3.04241605e-04]
# ye [ 2.51870188e-04 -7.71189085e-05 -3.33410980e-04]

