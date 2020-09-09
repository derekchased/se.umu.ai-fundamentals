"""
Stopwatch for Microsoft Robotic Developer Studio 4 via the Lokarria http interface.

This program will track the position of the robot and measure the time it takes to complete a given path.

Author: Erik Billing (billing@cs.umu.se)

Update by Ola Ringdahl 2019-09-19 adapted the code to the new class implementation of Lokarriaexample
"""
import math
from Robot import *
from Path import *

START_THRESHOLD = 0.1
PASS_THRESHOLD = 1.5
GOAL_THRESHOLD = 1


class Stopwatch:
    def __init__(self, path_name):
        p = Path(path_name)
        self._path = p.getPath()
        # make a robot to move around
        self._robot = Robot()
        self._startPosition = self._robot.getPosition()
        self._startTime = None
        self._goalTime = None

    def run(self):
        print('Waiting for robot to start the race...')
        while self.xy_distance(self._startPosition, self._robot.getPosition()) < START_THRESHOLD:
            time.sleep(0.01)
        self._startTime = time.time()
        print('GO!')
        totCount = float(len(self._path))
        for i, point in enumerate(self._path):
            self.pass_point(point)
            print('%.0f%% of the path completed: %.2f seconds' % (i / totCount * 100, time.time() - self._startTime))
        print('All points passed, looking for goal point...')
        self.pass_point(self._path[-1], GOAL_THRESHOLD)
        self._goalTime = time.time()
        print('Goal reached in %.2f seconds.' % (self._goalTime - self._startTime))

    # Wait until the robot has passed this point
    def pass_point(self, position, threshold=PASS_THRESHOLD):
        while self.xy_distance(position, self._robot.getPosition()) > threshold:
            time.sleep(0.01)

    # Check the distance between two xy-points
    def xy_distance(self, pos1, pos2):
        x1, y1 = pos1['X'], pos1['Y']
        x2, y2 = pos2['X'], pos2['Y']
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx * dx + dy * dy)


if __name__ == '__main__':
    stopwatch = Stopwatch('Path-around-table-and-back.json')
    stopwatch.run()
