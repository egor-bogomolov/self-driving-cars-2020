#!/usr/bin/env python
import rospy
import math
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn
from turtlesim.msg import Pose


"""
Based on the lecture and the following example:
http://wiki.ros.org/turtlesim/Tutorials/Moving%20in%20a%20Straight%20Line

The node creates multiple creepers that follow each other and the user's turtle.   
"""


class Creeper:

    def __init__(self, creeper_name, creeper_x, creeper_y, target_name, speed):
        self.spawn_turtle(creeper_name, creeper_x, creeper_y)
        self.speed = speed
        self.target_tracker = rospy.Subscriber('/' + target_name + '/pose', Pose, self.track_target)
        self.creeper_tracker = rospy.Subscriber('/' + creeper_name + '/pose', Pose, self.track_creeper)
        self.velocity_publisher = rospy.Publisher('/' + creeper_name + '/cmd_vel', Twist, queue_size=10)
        self.x = creeper_x
        self.y = creeper_y
        self.theta = 0

    def spawn_turtle(self, name, x, y):
        rospy.wait_for_service('/spawn')
        spawn = rospy.ServiceProxy('/spawn', Spawn)
        spawn(x, y, 0., name)

    def track_creeper(self, message):
        self.x = message.x
        self.y = message.y
        self.theta = message.theta

    def track_target(self, message):
        vel_msg = self.generate_message()
        t_x = message.x
        t_y = message.y
        dist = math.sqrt((t_x - self.x) ** 2 + (t_y - self.y) ** 2)
        if dist < 1e-2:
            return
        theta = math.atan2((t_y - self.y), (t_x - self.x))
        vel_msg.linear.x = min(self.speed, dist)
        vel_msg.angular.z = (theta - self.theta) * 10
        self.velocity_publisher.publish(vel_msg)

    def generate_message(self):
        vel_msg = Twist()
        vel_msg.linear.x = self.speed
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        return vel_msg


rospy.init_node('catch_node')
Creeper('creeper1', 3., 3., 'turtle1', 2.)
Creeper('creeper2', 2., 2., 'creeper1', 1.)
Creeper('creeper3', 1., 1., 'creeper2', 0.5)
Creeper('creeper4', 1., 2., 'turtle1', 0.5)
rospy.spin()

