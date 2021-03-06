""" Robot path following implementation based on the Pure Pursuit algorithm """

import sys
import math
from Robot import *
from Path import *
from ShowPath import *
from Stopwatch import *
import a1_functions as a1f
import matplotlib.pyplot as plt
import numpy as np

LOOK_AHEAD_DISTANCE = 1
# Robot should stop within 1 unit
GOAL_THRESHOLD = .5

LINEAR_SPEED_LEVEL_1 = 0.2
LINEAR_SPEED_LEVEL_2 = 0.25
LINEAR_SPEED_LEVEL_3 = 0.3
LINEAR_SPEED_LEVEL_4 = 0.35
LINEAR_SPEED_LEVEL_5 = 0.4

"""LINEAR_SPEED_LEVEL_1 = 0.4
LINEAR_SPEED_LEVEL_2 = 0.5
LINEAR_SPEED_LEVEL_3 = 0.6
LINEAR_SPEED_LEVEL_4 = 0.7
LINEAR_SPEED_LEVEL_5 = .8"""


class RobotController:

    # Constructor, takes path filename
    def __init__(self, path_name):
        self._robot = Robot()
        p = Path(path_name)
        path = p.getPath()
        
        # Convert path from json to numpy matrix
        self._path_matrix = a1f.conv_path_to_np(path)
        #print("Path", *((p["X"], p["Y"], p["Z"]) for p in path), sep="\n")
        #print("Path Matrix", self._path_matrix.shape, self._path_matrix)

        self._LOOK_AHEAD_DISTANCE = LOOK_AHEAD_DISTANCE
        self._sp = ShowPath(path)
        self._running = False
        #self._stop_watch = Stopwatch(path_name)


    def start_robot(self):
        self._running = True


    # Main method to determine and update robot's velocity and heading
    def take_step(self):
        #print("in take step, before stop watch run")
        #self._stop_watch.run()
        #print("in take step, after stop watch run")
        self._robot_position = self._robot.getPosition()

        # Convert robot's position to a numpy array
        self._robot_position_vector = a1f.conv_pos_to_np(self._robot_position)
        #print("robot position (raw json format):", self._robot_position)
        #print("robot position in numpy format:", self._robot_position_vector.shape, self._robot_position_vector)

        # Get distances from robot to each point
        self._robot_to_path_distances = a1f.compute_distances_vector_matrix(self._robot_position_vector, self._path_matrix)

        # Find furthest valid point
        goal_point_index = self._find_goal_point_index()

        # Get goal point as vector
        self._goal_point_coordinate_world = self._path_matrix[goal_point_index]
        #print("robot to path points distances", self._robot_to_path_distances.shape, self._robot_to_path_distances)
        ##print("goal point index: ", goal_point_index)
        ##print("goal point coordinate: ", self._goal_point_coordinate_world.shape, self._goal_point_coordinate_world)

        #fig, ax = plt.subplots(1, 1)
        #ax.plot(np.arange(len(self._robot_to_path_distances)), self._robot_to_path_distances)  # plot the path
        #plt.show(block=True)

        #self._goal_point_coordinate_world_2D = a1f.conv_pos_to_np_2D(self._robot_position)

        # Get goal point x and y
        goal_point_x_WCS = self._goal_point_coordinate_world[0]
        goal_point_y_WCS = self._goal_point_coordinate_world[1]

        # Get robot's global x and y coordinate
        robot_position_x_WCS = self._robot_position_vector[0]
        robot_position_y_WCS = self._robot_position_vector[1]

        # Get robot's current heading
        psi = self._robot.getHeading()
        ##print("Robot heading in WCS: ", psi)

        # Convert goal point in world coordinates to robot's local coordinates
        goal_point_x_RCS =  (goal_point_x_WCS - robot_position_x_WCS) * cos(psi) + (goal_point_y_WCS - robot_position_y_WCS) * sin(psi)
        goal_point_y_RCS = -(goal_point_x_WCS - robot_position_x_WCS) * sin(psi) + (goal_point_y_WCS - robot_position_y_WCS) * cos(psi)

        # Get robot local heading towards goal point
        gp_angle_RCS = atan2(goal_point_y_RCS, goal_point_x_RCS)

        # Get heading in local degrees
        gp_abs_angle_RCS_degree = abs(gp_angle_RCS) * 180 / math.pi

        # Choose linear speed based on degree of turning angle (tighter angle, slower speed)
        linear_speed = 0
        if (gp_abs_angle_RCS_degree <= 10):
            linear_speed = LINEAR_SPEED_LEVEL_5
        elif (10 < gp_abs_angle_RCS_degree and gp_abs_angle_RCS_degree <= 20):
            linear_speed = LINEAR_SPEED_LEVEL_4
        elif (20 < gp_abs_angle_RCS_degree and gp_abs_angle_RCS_degree <= 30):
            linear_speed = LINEAR_SPEED_LEVEL_3
        elif (30 < gp_abs_angle_RCS_degree and gp_abs_angle_RCS_degree <= 45):
            linear_speed = LINEAR_SPEED_LEVEL_2
        elif (45 < gp_abs_angle_RCS_degree):
            linear_speed = LINEAR_SPEED_LEVEL_1

        #linear_speed = LINEAR_SPEED_LEVEL_5
        #print("Goal point angle in RCS: ", gp_angle_RCS * 180 / math.pi)
        #print("\ngp_abs_angle_RCS_degree: ", gp_abs_angle_RCS_degree)

        # Calculate turn rate
        g = 2 * goal_point_y_RCS / LOOK_AHEAD_DISTANCE**2
        turn_rate = linear_speed * g
        #print("turn_rate: ", turn_rate)

        # Update robot speed and turn rate
        self._robot.setMotion(linear_speed, turn_rate)

        # Shorten the path matrix
        self._path_matrix = self._path_matrix[goal_point_index:,:]
        #print(len(self._path_matrix))

        # Plot the robots point
        self._sp.update(self._robot.getPosition(), self._goal_point_coordinate_world)

        # Determine if robot should stop
        #print("len(self._path_matrix)",len(self._path_matrix))
        #print("self._robot_to_path_distances[0]",self._robot_to_path_distances[0])
        #print("GOAL_THRESHOLD",GOAL_THRESHOLD)
        #print("self._robot_to_path_distances[0] < GOAL_THRESHOLD",self._robot_to_path_distances[0] < GOAL_THRESHOLD)
        if (len(self._path_matrix) == 1 and self._robot_to_path_distances[0] < GOAL_THRESHOLD):
            self._running = False

    #def get_path_length(self):
    #    return len(self._path_matrix)

    def stop_running(self):
        self._robot.setMotion(0.0,0.0)


    def get_running_status(self):
        return self._running

    # Show plot and keep visible when python execution ends
    def pause_plot(self):
        self._sp.pause_the_plot()


    #def print_path_length(self):
    #    ShowPath(self._path)
    #    print("Path length = " + str(len(self._path)))
    #    for i in range(len(path)):
    #        print("Point number %d on the path!" )

    def _find_goal_point_index(self):

        # initialize to first point in the path
        goal_point_index = 0

        # iterate through points along the path, from first to last
        # if a point is <= look ahead distance, choose as next point
        # if a point is > look ahead distance break out of loop and return the index
        for i, j in enumerate(self._robot_to_path_distances):
            if j <= LOOK_AHEAD_DISTANCE:
                goal_point_index = i
            else:
                break
        return goal_point_index


if __name__ == "__main__":

    # Filename of the path is passed in through the command line argument
    robotController = RobotController(sys.argv[1])

    # Available paths:
    # Path-around-table-and-back.json
    # Path-around-table.json
    # Path-to-bed.json
    # Path-from-bed.json

    # Start robot driving
    robotController.start_robot()

    # Update robot heading and velocity every .35 seconds, while status is True
    while robotController.get_running_status() == True:
        time.sleep(0.1)
        robotController.take_step()

    # Stop the robot
    robotController.stop_running()

    # Show plot and keep visible when python execution ends
    robotController.pause_plot()