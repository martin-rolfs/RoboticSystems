
from robotic_systems.robot import Robot

# ROS shit
from rclpy.node import Node

from lbr_fri_idl.msg import LBRJointPositionCommand, LBRState

class RobotNode(Node):
    def __init__(self, nodeName: str="kuka_interface", robotModel: Robot=None):
        super.__init__("kuka_interface")
        self.robotModel = robotModel
        self.get_logger().info("Robot Node has been started.")
        self.joint_state_publisher = self.create_publisher(LBRJointPositionCommand, "command/joint_position", 10)

        pass