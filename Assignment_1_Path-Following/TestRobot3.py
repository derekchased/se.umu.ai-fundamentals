from Robot import *
from Path import *
from ShowPath import *


# load a path file
p = Path("Path-around-table.json")
path = p.getPath()

# plot the path
sp = ShowPath(path)

print("Path length = " + str(len(path)))
print("First point = " + str(path[0]['X']) + ", " + str(path[0]['Y']))

# make a robot to move around
robot = Robot()

# move the robot
robot.setMotion(0.2, 0.2)

for i in range(10):
    time.sleep(0.5)
    print("pos, heading")
    print(robot.getPosition())
    print(robot.getHeading())

    # Plot the current position and the look-ahead point:
    look_ahead_point = [3+i/5,4+i/5] #just a dummy point that moves 
    sp.update(robot.getPosition(), look_ahead_point)

echoes = robot.getLaser()
print(echoes)

robot.setMotion(0, 0)
