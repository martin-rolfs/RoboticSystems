# Robotic Systems Package

This package defines basic classes that can be used to model robots.

ROS dependencies are not listed in `setup.py`. Use `robot_node` module only in ROS2 python environment. 

### Adding ROS Interfaces

WIP: Adding a ROS interface that uses the LBR stack to communicate with robot. This should be tested in correspondance with ROS simulation environemnt (e.g. Gazebo). 

- Robot class as interface to robot
- StereoCam class as interface to stereoendoscope
    - Does this run with ROS? Retrieval not clear in this case