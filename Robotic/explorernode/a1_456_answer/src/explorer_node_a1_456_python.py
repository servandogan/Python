import rospy
import sys

## This is needed for the data structure containing the motor command.
from geometry_msgs.msg import Twist
## This is needed for the data structure containing the laser scan
from sensor_msgs.msg import LaserScan
## This is needed for the data structure containing the map.
from nav_msgs.msg import OccupancyGrid

## The following function is a "callback" function that is called back whenever a new laser scan is available.
## That is, this function will be called for every new laser scan.

def laser_callback(data):
    ## Lets fill a twist message for motor command
    motor_command = Twist()
    
    motor_command.linear.x = 0
    motor_command.angular.z = 0
    
    global motor_command_publisher
    
    #for i in range(len(data.ranges)):
    #	print(str(i) + ":" + str(data.ranges[i]))
    
    # Variables to check whether front/rear is occupied or not
    frontocc = False
    rearocc = False
    
    # How many degrees the bot will check for collisions
    numb = 45
    
    for i in range(numb):
    	frontocc = frontocc or data.ranges[i - numb // 2] < 0.5
    	rearocc = rearocc or data.ranges[i - numb // 2 + 270] < 0.4
    
    # Variable to keep track of the position of the furthest object
    max_angle = 0
    
    # Find the furthest object from the bot.
    for i in range(91):
    	if(data.ranges[max_angle] < data.ranges[i]):
    		max_angle = i
    for i in range(91):
    	if(data.ranges[max_angle] < data.ranges[359 - i]):
    		max_angle = 359 - i
    
    # Debug print to check which angle has the furthest object
    # print(str(max_angle) + ":" + str(data.ranges[max_angle]))
    
    # If front is occupied, go back. If rear is also occupied then try to turn around and get out.
    if frontocc:
    		motor_command.linear.x = -1
    		motor_command.angular.z = 0
    		if rearocc:
    			motor_command.angular.z = -1
    
    # If front is not occupied and the furthest object is in 5 degrees in front of the bot, go straight ahead.
    elif(max_angle < 5 or max_angle > 355):
    		motor_command.linear.x = 1
    		motor_command.angular.z = 0
    
    # If the furthest object is on the right, turn right
    elif(max_angle >= 180):
    		motor_command.linear.x = 0.5
    		motor_command.angular.z = -0.7

    # If the furthest object is on the left, turn left
    elif(max_angle < 180):
    		motor_command.linear.x = 0.5
    		motor_command.angular.z = 0.7

    # In any other situation(which shouldn't happen) turn around aimlessly hoping for the situation to change
    else:
    		motor_command.linear.x = 0
    		motor_command.angular.z = 1
    
    # Send motor info
    motor_command_publisher.publish(motor_command)


def map_callback(data):
    chatty_map = False
    
    if chatty_map:
        print ("-------MAP---------")
        ## Here x and y has been incremented with five to make it fit in the terminal
        ## Note that we have lost some map information by shrinking the data
        for x in range(0,data.info.width-1,5):
            for y in range(0,data.info.height-1,5):
                index = x+y*data.info.width
                if data.data[index] > 50:
                    ## This square is occupied
                    sys.stdout.write('X')
                elif data.data[index] >= 0:
                    ## This square is unoccupied
                    sys.stdout.write(' ')
                else:
                    sys.stdout.write('?')
            sys.stdout.write('\n')
        sys.stdout.flush()
        print ("-------------------")
    
def explorer_node():
    rospy.init_node('explorer')
    
    ## Here we declare that we are going to publish "Twist" messages to the topic /cmd_vel_mux/navi. It is defined as global because we are going to use this publisher in the laser_callback.
    global motor_command_publisher
    motor_command_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
    
    ## Here we set the function laser_callback to recieve new laser messages when they arrive
    rospy.Subscriber("/scan", LaserScan, laser_callback, queue_size = 1000)
    
    ## Here we set the function map_callback to recieve new map messages when they arrive from the mapping subsystem
    rospy.Subscriber("/map", OccupancyGrid, map_callback, queue_size = 1000)
    
    ## spin is an infinite loop but it lets callbacks to be called when a new data available. That means spin keeps this node not terminated and run the callback when nessessary. 
    rospy.spin()
    
if __name__ == '__main__':
    explorer_node()
