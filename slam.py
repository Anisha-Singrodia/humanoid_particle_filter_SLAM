# Pratik Chaudhari (pratikac@seas.upenn.edu)

import os, sys, pickle, math
from copy import deepcopy
from matplotlib import pyplot
from matplotlib import colors

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

        s.free_prob_thresh = 0.2
        s.log_odds_free = np.log(s.free_prob_thresh/(1-s.free_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        # -20 to 20 i.e 0 to 40
        x = np.clip(x, s.xmin, s.xmax)
        y = np.clip(y, s.ymin, s.ymax)
        output = np.zeros((2, len(x)))
        cumm_arr = np.zeros((s.szx))
        for i in range(1, s.szx):
            # if i==0:
            #     cumm_arr[i] = 40/s.szx
            # else:
            cumm_arr[i] = cumm_arr[i-1] + 40/s.szx
        # print(cumm_arr)
        for i in range(len(x)):
            for j in range(s.szx):
                if cumm_arr[j]-20 > x[i]:
                    break
            output[0, i] = j-1
            for j in range(s.szx):
                if cumm_arr[j]-20 > y[i]:
                    break
            output[1, i] = j-1
        return output

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = 1e-8*np.eye(3)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)
        s.particle_posn = None

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closest idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX
        n = p.shape[1]
        
        cumm_w = np.zeros(w.shape)
        new_p = np.zeros(p.shape)
        cumm_w[0,0] = w[0,0]
        for i in range(1,n):
            cumm_w[0,i] = cumm_w[0,i-1] + w[0,i]
        for i in range(n):
            rand_w = np.random.uniform(0,1)
            for j in range(n):
                if cumm_w[0,j] > rand_w:
                    break
            new_p[:, i] = p[:, j-1]
        w[0, :] = 1/n
        print(new_p)
        return new_p, w

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        d = np.clip(d, s.lidar_dmin, s.lidar_dmax)
        assert (d[:] >= s.lidar_dmin).all() and (d[:] <= s.lidar_dmax).all()
        # print("assertion complete")

        # 1. from lidar distances to points in the LiDAR frame
        num_rays = len(angles)
        coords = np.zeros((4, num_rays))
        coords[0, :] = d[:]*np.cos(angles[:])
        coords[1, :] = d[:]*np.sin(angles[:])
        coords[2, :] = np.zeros_like(coords[0])
        coords[3, :] = np.ones_like(coords[0])
        # 2. from LiDAR frame to the body frame
        neck_rot = euler_to_se3(0, 0, neck_angle, np.array([0,0,0]))
        head_rot = euler_to_se3(0, head_angle, 0, np.array([0,0,0]))
        body_to_head = np.dot(neck_rot, head_rot)
        head_to_lidar = euler_to_se3(0, 0, 0, np.array([0,0,0.15])) #
        lidar_to_body = np.dot(body_to_head, head_to_lidar)
        coords_body = np.matmul(lidar_to_body, coords)

        # 3. from body frame to world frame

        body_to_world = euler_to_se3(0, 0, p[2], np.array([p[0], p[1], s.head_height]))
        coords_world = np.matmul(body_to_world, coords_body)
        return coords_world[0:2, :]

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)
        #### TODO: XXXXXXXXXXX
        idx_t = np.argmin(np.abs(s.joint['t']-t))
        idx_t = t
        pt = s.lidar[idx_t]['xyth']
        # print("pt :", pt)
        pt_ = s.lidar[idx_t-1]['xyth']
        # exit()
        return smart_minus_2d(pt, pt_)
         
    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        control = s.get_control(t)
        # print("control :", control)
        updated_p = np.zeros_like(s.p)
        p = s.p
        if s.n == 1:
            updated_p = updated_p.reshape((3,-1))
            p = p.reshape((3,-1))
        # print(updated_p)
        for i in range(s.n):
            updated_p[:, i] = smart_plus_2d(p[:, i], control) + np.random.multivariate_normal(np.zeros((3,)), s.Q)
        if s.n == 1:
            updated_p = updated_p.reshape((3))
        s.p = updated_p

    @staticmethod
    def update_weights(w, obs_logp):

        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        weights = np.log(w) + obs_logp
        weights = np.exp(weights - slam_t.log_sum_exp(weights))
        return weights

    def get_free_cells(s, occ_cells, particle_posn):
        free_cells = particle_posn

        for i in occ_cells:
            x = np.linspace(particle_posn[0], i[0], int(np.linalg.norm(particle_posn - i)), endpoint=False, dtype=int)
            y = np.linspace(particle_posn[1], i[1], int(np.linalg.norm(particle_posn - i)), endpoint=False, dtype=int)
            free_cells = np.hstack((free_cells, np.vstack((x.T, y.T))))
        free_cells = np.unique(free_cells, return_index=False, axis=1)
        free_cells = free_cells.astype(int)
        return free_cells
        


        # free_cells = cur_loc
        # x_s = cur_loc[0]
        # y_s = cur_loc[1]

        # for i in obs_loc.T:
        #     x_o = i[0]
        #     y_o = i[1]
        #     dir = i - cur_loc.T
        #     dist = int(np.linalg.norm(dir))
        #     x = np.linspace(x_s, x_o, dist, endpoint=False, dtype=int)
        #     y = np.linspace(y_s, y_o, dist, endpoint=False, dtype=int)
        #     new_free_cells = np.vstack((x.T, y.T))
        #     print("new_free_cells: ", new_free_cells.shape)
        #     free_cells = np.hstack((free_cells, new_free_cells))
        #     print("free_cells: ", free_cells.shape)
        # free_cells = np.unique(free_cells, return_index=False, axis=1)

        # return free_cells






    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        joint_idx = s.find_joint_t_idx_from_lidar(t)
        head_angle = s.joint['head_angles'][0][joint_idx]
        neck_angle = s.joint['head_angles'][1][joint_idx]
        obs_log_p = np.zeros(s.n)
        lidar_cells = np.zeros((s.n, 2, 1081))
        for i in range(s.n):
            if s.n == 1:
                points = s.rays2world(s.p, s.lidar[t]['scan'], head_angle, neck_angle, s.lidar_angles)
            else:
                points = s.rays2world(s.p[:, i], s.lidar[t]['scan'], head_angle, neck_angle, s.lidar_angles)
            map = s.map.grid_cell_from_xy(points[0], points[1]).astype(int)
            lidar_cells[i] = map
            obs_log_p[i] = np.sum(s.map.cells[map[0], map[1]])
        lidar_cells = lidar_cells.astype(int)
        s.w = s.update_weights(s.w, obs_log_p)
        p_highest_weight = np.argmax(s.w)
        if s.n == 1:
            p_highest = s.p
        else:
            p_highest = s.p[:, p_highest_weight]
        s.particle_posn = p_highest
        particle_cells = s.map.grid_cell_from_xy(np.array([p_highest[0]]), np.array([p_highest[1]]))
        s.particle_on_map = particle_cells
        # print(particle_cells)
        occupied = lidar_cells[p_highest_weight]
        # free = s.get_free_cells(occupied, particle_cells)
        # print(occupied[0])
        s.map.log_odds[:,:] += s.lidar_log_odds_free
        s.map.log_odds[occupied[0], occupied[1]] += s.lidar_log_odds_occ - s.lidar_log_odds_free
        # s.map.log_odds[free[0], free[1]] += s.lidar_log_odds_free
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)
        # occ_cells = np.where(s.map.log_odds >= s.map.log_odds_thresh)
        occ_cells = occupied
        # free_cells = np.where(s.map.log_odds <= s.map.log_odds_free)
        # free_cells = free
        # print(free_cells)
        # s.map.cells = 0*np.ones_like(s.map.cells)
        # print(s.map.cells.shape)
        s.map.cells[occ_cells[0], occ_cells[1]] = 1
        # s.map.cells[free_cells[0], free_cells[1]] = 0
        # pyplot.figure(figsize=(5,5))
        # colormap = colors.ListedColormap(["grey","white","black"])
        # pyplot.imshow(s.map.cells, cmap=colormap)
        # pyplot.show()
        

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')