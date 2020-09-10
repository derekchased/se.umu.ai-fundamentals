""" Robot path following implementation based on the Pure Pursuit algorithm """

import math
from Robot import *
from Path import *
from ShowPath import *
from Stopwatch import *
import a1_functions as a1f
import matplotlib.pyplot as plt
import numpy as np

LOOK_AHEAD_DISTANCE = 1
GOAL_THRESHOLD = 0.1

class RobotController:
    def __init__(self, path_name):
        self._robot = Robot()
        p = Path(path_name)
        path = p.getPath()
        self._path_matrix = a1f.conv_path_to_np(path)
        #print("Path", *((p["X"], p["Y"], p["Z"]) for p in path), sep="\n")
        #print("Path Matrix", self._path_matrix.shape, self._path_matrix)
        self._LOOK_AHEAD_DISTANCE = LOOK_AHEAD_DISTANCE
        self._sp = ShowPath(path)
        self._running = False
        #self._stop_watch = Stopwatch(path_name, self._robot)


    def start_robot(self):
        self._running = True


    def take_step(self):
        #print("in take step, before stop watch run")
        #self._stop_watch.run()
        #print("in take step, after stop watch run")
        self._robot_position = self._robot.getPosition()
        self._robot_position_vector = a1f.conv_pos_to_np(self._robot_position)
        #print("robot position (raw json format):", self._robot_position)
        #print("robot position in numpy format:", self._robot_position_vector.shape, self._robot_position_vector)
        self._robot_to_path_distances = a1f.get_distances_optimal1(self._robot_position_vector, self._path_matrix)
        goal_point_index = self._find_goal_point_index()
        self._goal_point_coordinate_world = self._path_matrix[goal_point_index]
        #print("robot to path points distances", self._robot_to_path_distances.shape, self._robot_to_path_distances)
        print("goal point index: ", goal_point_index)
        print("goal point coordinate: ", self._goal_point_coordinate_world)

        #fig, ax = plt.subplots(1, 1)
        #ax.plot(np.arange(len(self._robot_to_path_distances)), self._robot_to_path_distances)  # plot the path
        #plt.show(block=True)

        #self._goal_point_coordinate_world_2D = a1f.conv_pos_to_np_2D(self._robot_position)

        goal_point_x_WCS = self._goal_point_coordinate_world[0]
        goal_point_y_WCS = self._goal_point_coordinate_world[1]

        robot_position_x_WCS = self._robot_position_vector[0]
        robot_position_y_WCS = self._robot_position_vector[1]

        psi = self._robot.getHeading()
        print("Robot heading in WCS: ", psi)

        goal_point_x_RCS =  (goal_point_x_WCS - robot_position_x_WCS) * cos(psi) + (goal_point_y_WCS - robot_position_y_WCS) * sin(psi)
        goal_point_y_RCS = -(goal_point_x_WCS - robot_position_x_WCS) * sin(psi) + (goal_point_y_WCS - robot_position_y_WCS) * cos(psi)

        gp_angle = atan2(goal_point_y_RCS, goal_point_x_RCS)
        print("Goal point angle in RCS: ", gp_angle * 180 / math.pi)

        g = 2 * goal_point_y_RCS / LOOK_AHEAD_DISTANCE**2
        linear_speed = 0.4
        turn_rate = linear_speed * g
        self._robot.setMotion(linear_speed, turn_rate)
        self._path_matrix = self._path_matrix[goal_point_index:,:]
        print(len(self._path_matrix))
        self._sp.update(self._robot.getPosition(), self._goal_point_coordinate_world)
        if (len(self._path_matrix) == 1 and self._robot_to_path_distances[0] < GOAL_THRESHOLD):
            self._running = False

    #def get_path_length(self):
    #    return len(self._path_matrix)

    def stop_running(self):
        self._robot.setMotion(0.0,0.0)


    def get_running_status(self):
        return self._running

    def pause_plot(self):
        self._sp.pause_the_plot()


    #def print_path_length(self):
    #    ShowPath(self._path)
    #    print("Path length = " + str(len(self._path)))
    #    for i in range(len(path)):
    #        print("Point number %d on the path!" )

    def _find_goal_point_index(self):
        goal_point_index = 0
        for i, j in enumerate(self._robot_to_path_distances):
            if j <= LOOK_AHEAD_DISTANCE:
                goal_point_index = i
            else:
                break
        return goal_point_index


if __name__ == "__main__":
    #robotController = RobotController('Path-around-table-and-back.json')
    #robotController = RobotController('Path-around-table.json')
    #robotController = RobotController('Path-to-bed.json')
    robotController = RobotController('Path-from-bed.json')
    robotController.start_robot()
    while robotController.get_running_status() == True:
        time.sleep(0.35)
        robotController.take_step()

    robotController.stop_running()
    robotController.pause_plot()