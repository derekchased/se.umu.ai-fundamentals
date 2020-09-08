

import numpy as np
from collections import Counter # for counting class of neighbors

def get_distances_optimal1(robot_pos, path_matr):
    """ Get the distances between Xtrain and Zpredict
    Note:
        1. Optimization level - highly optimized!
        
        Algorithm was found https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        It is a no loop solution, which means it can handle the matrices in
        one line of code rather than having to iterate over the rows of one
        matrix
        
        2. I have removed the square root to further optimize the calculation.
        This is ok, because I do not need the actual distance between points. 
        I just need a list of the relative distances. Since the sq root is 
        monotonic, it preserverses the order.         
    """
    # Convert position vector into a matrix
    robot_pos_matr = np.full(path_matr.shape,robot_pos)

    print("robot_pos_matr",robot_pos_matr.shape,robot_pos_matr)
    dists = np.sqrt(-2 * np.dot(path_matr, robot_pos_matr.T) + np.sum(robot_pos_matr**2, axis=1) + np.sum(path_matr**2, axis=1)[:, np.newaxis])
    #dists = -2 * np.dot(path_matr, robot_pos_matr.T) + np.sum(robot_pos_matr**2, axis=1) + np.sum(path_matr**2, axis=1)[:, np.newaxis]
    return dists[:,0]

def calc_distances_robot_to_path_points(robot_pos, path):
    """ Get the distances between robot_pos_matr and Y
    Note:
        Algorithm was found https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        - It is a no loop solution, which means it can handle the matrices in
        one line of code rather than having to iterate over the rows of one
        matrix
        - Originally found in a previous Intro ML course @ LNU.SE
        
        2. I have removed the square root to further optimize the calculation.
        This is ok, because I do not need the actual distance between points. 
        I just need a list of the relative distances. Since the sq root is 
        monotonic, it preserverses the order.         
    """

    # Convert position into a matrix
    robot_pos_matr = np.full(path_matr.shape,robot_pos_np)

    #dists = np.sqrt(-2 * np.dot(robot_pos_matr, X.T) + np.sum(X**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis])
    dists = -2 * np.dot(Y, X.T) + np.sum(X**2, axis=1) + np.sum(Y**2, axis=1)[:, np.newaxis]
    return dists

# Take a single position from provided path file and convert it to a numpy array 
def conv_pos_to_np(pos):
    return np.array([pos["X"], pos["Y"], pos["Z"]])

def conv_pos_to_np_2D(pos):
    return np.array([pos["X"], pos["Y"]])

# Take a path from provided path files and convert it to a numpy array
def conv_path_to_np(p):
    return np.array(  [ conv_pos_to_np(pos) for pos in p  ])

def go_to_point(robot, point):
    pass

def sort_closest_path_points(robot_path_distances):
    return np.argsort(robot_path_distances)