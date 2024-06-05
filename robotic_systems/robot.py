import numpy as np
import pickle # to save object as binary file
import sympy as sp

from math import sin, cos

from os.path import join

from robotic_systems.pose import Transform

class Robot:    
    def __init__(self, initConfig: np.array, dh_theta: np.array, dh_d: np.array, dh_a: np.array, dh_alpha: np.array, upperLimit: np.array, lowerLimit: np.array, jointTypes: list=None, position: np.array=np.zeros((3, 1))):
        """Creates an robot model specified by its DH parameters. The joint types define what DH parameter the configuration is influencing. 
        In this case, the value provided for the parameters is added to the joint value. 

        Note, that the DH parameters are NOT expected to be the modified DH parameters.

        Args:
            initConfig (np.array): The initial configuration of the robot. The length of this array determines the number of joints.
            dh_theta (np.array): The rotation around the last z axis to align both x axes.
            dh_d (np.array): The translation along the last z axis to the next x axis.
            dh_a (np.array): The translation along the next x axis to the origin of the last coordinate system.
            dh_alpha (np.array): The rotation around the next x axis to align both z axes.
            upperLimit (np.array): The upper joint limit.
            lowerLimit (np.array): The lower joint limit.
            jointTypes (list): A list of str specifying each joint type. Allowed str are: 'theta', 'd', 'a' and 'alpha'. Each corresponds to the appropriate DH parameter. If None, it defaults to revolute joints.
            position (np.array, optional): The position of the base frame of the robot in world space. Defaults to np.zeros((3, 1)).

        Raises:
            ValueError: If the number of DH parameters or joint limits does not match the configuration an error is thrown.
        """
        self.config = initConfig
        self.nJoints = len(initConfig)

        if len(upperLimit) < self.nJoints or len(lowerLimit) < self.nJoints or len(dh_theta) < self.nJoints or len(dh_a) < self.nJoints or len(dh_alpha) < self.nJoints or len(dh_d) < self.nJoints:
           raise ValueError("Please specify enough joint limits and DH Parameters, according to the initial configuration!")

        self.upperLimit = upperLimit
        self.lowerLimit = lowerLimit
        self.dh = {
            'theta': dh_theta,
            'd': dh_d,
            'a': dh_a,
            'alpha': dh_alpha
        }

        if jointTypes is None:
            jointTypes = ['theta'] * self.nJoints
        self.jointTypes = jointTypes
        self.position = position # cartesian position in world space
        self.J = None

    def createJacobian(self, toolTransformation: np.array=np.eye(4)) -> sp.Matrix:   
        """Creates the geometric jacobian for the specified dh parameters.

        Args:
            toolTransformation (np.array, optional): The transformation matrix from the TFC to the TCP. Defaults to np.eye(4).

        Returns:
            sp.Matrix: Returns the symbolic jacobian with the joint values (j0, j1, ... , jn) as symbolics.
        """          
        T = sp.eye(4) # unit matrix
        # write position to first transformation
        T[0:3, 3] = self.position

        # save specific entries of matrix
        z = [sp.Matrix([[0], [0], [1]])]
        p = [sp.Matrix([[0], [0], [0]])]

        # create transformation matrices
        for jIndex in range(self.nJoints):
            T = T * self.getTransformMatrixSymbolic(jIndex, name=f"j{jIndex}")

            if jIndex == self.nJoints - 1:
                T = T * toolTransformation

            z.append(T[0:3, 2])
            p.append(T[0:3, 3])

        P = T[0:3, 3]

        # Base jacobian
        J = sp.Matrix([[z[0].cross(P - p[0])], [z[0]]])

        for jIndex in range(self.nJoints-1):
            newCol = sp.Matrix([[z[jIndex+1].cross(P - p[jIndex+1])], [z[jIndex+1]]])
            J = J.row_join(newCol)

        print(f"Jacobian Size: {J.rows}x{J.cols}")
        print("Simplifying Jacobian... This may take a while.")
        J = sp.nsimplify(J,tolerance=1e-10,rational=True)

        return J

    def getTCP(self, config: np.array=None, toolTransformation: np.array=np.eye(4)) -> np.ndarray:
        """Calculates the homogenous transformation matrix that describes the pose at the TCP.

        Args:
            config (np.array, optional): The configuration of the robot. If it is None, the attribute is used. Defaults to None.
            toolTransformation (np.array, optional): The transformation matrix from the TFC to the TCP. Defaults to np.eye(4).

        Returns:
            np.ndarray: Returns the transformation matrix as numpy array.
        """        
        if config is None:
            config = self.config
    
        T = np.eye(4)
        T[0:3, 3] = self.position.reshape(3,)

        # create transformation matrices
        for jIndex in range(self.nJoints):
            T = T @ self.getTransformMatrix(jIndex, config[jIndex])

            if jIndex == self.nJoints - 1:
                T = T @ toolTransformation

        return T

    def convertToTCS(self, pose: np.array, config: np.array=None, tooltransformationMatrix: np.array=np.eye(4)) -> np.array:
        """Perform basis transformation from worldspace pose to tool coordinate system (TCS).

        Args:
            pose (np.array): The pose  as a 4x4 homogenous transformation matrix to transform.
            config (np.array, optional): The configuration of the robot. If it is None, the attribute is used. Defaults to None.
            toolTransformation (np.array, optional): The transformation matrix from the TFC to the TCP. Defaults to np.eye(4).

        Returns:
            np.ndarray: Returns the transformation matrix as numpy array.
        """        
        if config is None:
            config = self.config

        # get complete transformation matrix
        baseToTCP = self.getTCP(config, tooltransformationMatrix)

        poseInTCS = np.linalg.inv(baseToTCP) @ pose
        rotInTCS = Transform.invertHomogenousTransform(baseToTCP) @ pose
        poseInTCS[0:3, 0:3] = rotInTCS[0:3, 0:3]

        return poseInTCS

    def getTransformMatrixSymbolic(self, jointIndex: int, name: str="j") -> sp.Matrix:
        # get base values
        theta = self.dh['theta'][jointIndex]
        alpha = self.dh['alpha'][jointIndex]
        a = self.dh['a'][jointIndex]
        d = self.dh['d'][jointIndex]

        j = sp.Symbol(name, real=True)

        # modify according to type with symbolic
        if self.jointTypes[jointIndex] == 'theta':
            theta = j + theta
        elif self.jointTypes[jointIndex] == 'a':
            a = j + a
        elif self.jointTypes[jointIndex] == 'd':
            d = j + d
        elif self.jointTypes[jointIndex] == 'alpha':
            alpha = j + alpha

        # return symbolic matrix
        return sp.Matrix([[sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
                          [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)], 
                          [0,              sp.sin(alpha),                sp.cos(alpha),               d], 
                          [0,              0,                            0,                           1]])
    
    def getTransformMatrix(self, jointIndex: int, j: float) -> np.ndarray:
        # get base values
        theta = self.dh['theta'][jointIndex]
        alpha = self.dh['alpha'][jointIndex]
        a = self.dh['a'][jointIndex]
        d = self.dh['d'][jointIndex]

        # modify with joint value accordingly
        if self.jointTypes[jointIndex] == 'theta':
            theta = j + theta
        elif self.jointTypes[jointIndex] == 'a':
            a = j + a
        elif self.jointTypes[jointIndex] == 'd':
            d = j + d
        elif self.jointTypes[jointIndex] == 'alpha':
            alpha = j + alpha    

        return np.array([[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                         [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)], 
                         [0,           sin(alpha),             cos(alpha),            d], 
                         [0,           0,                      0,                     1]])
    
    def constraintJointLimits(self, config: np.array=None) -> np.array:
        """Constraints the configuration to the provided joint limits.

        Args:
            config (np.array, optional): The configuration to constraint, if it is None, the attribute is used. Defaults to None.

        Returns:
            np.array: The new constrained configuration.
        """        
        if config is None:
            config = self.config

        constrainedConfig = np.minimum(config, self.upperLimit)
        constrainedConfig = np.maximum(constrainedConfig, self.lowerLimit)

        return constrainedConfig

    def getCurrentJacobian(self, config: np.array=None) -> sp.Matrix:
        """Determines a Jacobian matrix for a specific configuration.

        Args:
            config (np.array, optional): The configuration to be used. If it is None the attribute is used. Defaults to None.

        Raises:
            ValueError: Raises an error if the `J` attribute has not been written to.

        Returns:
            sp.Matrix: Returns the jacobian as a symbolic matrix. No symbolics should be included anymore. It can be cast as a numpy array with `np.array(J).astype(np.float64)` 
        """        
        if self.J is None:
            raise ValueError("The Jacobian has not been generated yet. Please use `createJacobian` to write to field `J`.")
        if config is None:
            config = self.config

        values = {}
        for jIndex in range(self.nJoints):
            values.update({sp.Symbol(f"j{jIndex}", real=True): config[jIndex]})

        spefJ = self.J.subs(values)
        return spefJ

    def saveAsBinary(self, directory: str):
        """Save the constructed robot as a binary file. Can be used to avoid lengthy jacobian simplification.

        Args:
            directory (str): The path to the directory where the binary will be saved as 'robot.pickle'.
        """
        with open(join(directory, "robot.pickle"), 'wb') as outFile:
            outFile.write(pickle.dumps(self))

    @staticmethod
    def loadRobotBinary(directory: str) -> "Robot":
        """Construct a robot object with a binary file.

        Args:
            directory (str): The path to the 'robot.pickle' file.   

        Returns:
            Robot: The loaded robot.
        """
        robot = None
        with open(join(directory, "robot.pickle"), 'rb') as inFile:
            robot = pickle.loads(inFile.read())

        return robot
