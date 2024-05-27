import numpy as np

class Transform:
    def __init__(self, eulerAngles: np.array, position: np.array, transformationMatrix: np.array):
        if not transformationMatrix is None:
            self.H = transformationMatrix
            self.eulerAngles = Transform.rot2eul(self.H[0:3, 0:3])
            self.position = self.H[0:3, 3]
        elif not eulerAngles is None and not position is None:        
            self.eulerAngles = eulerAngles
            self.position = position
            self.H = np.eye(4)
            self.H[0:3, 0:3] = Transform.eul2rot(self.eulerAngles)
            self.H[0:3, 3] = self.position 
        else:
            raise ArithmeticError("Euler angles and the position or the transformation matrix has to be provided.")

    def __str__(self) -> str:
        return str(f"""⌈ {round(self.H[0,0], 2)}  {round(self.H[0,1], 2)}  {round(self.H[0,2], 2)}  {round(self.H[0,3], 2)} ⌉
                   | {round(self.H[1,0], 2)}  {round(self.H[1,1], 2)}  {round(self.H[1,2], 2)}  {round(self.H[1,3], 2)} |
                   | {round(self.H[2,0], 2)}  {round(self.H[2,1], 2)}  {round(self.H[2,2], 2)}  {round(self.H[2,3], 2)} |
                   ⌊ {round(self.H[3,0], 2)}  {round(self.H[3,1], 2)}  {round(self.H[3,2], 2)}  {round(self.H[3,3], 2)} ⌋""")

    def setTransformMatrix(self, transformMatrix: np.array):
        self.H = transformMatrix
        self.eulerAngles = Transform.rot2eul(self.H[0:3, 0:3])
        self.position = self.H[0:3, 3]

    def setPosition(self, position: np.array):
        self.position = position
        self.H[0:3, 3] = position

    def setEulerAngles(self, eulerAngles: np.array):
        self.eulerAngles = eulerAngles
        self.H[0:3, 0:3] = Transform.eul2rot(self.eulerAngles)

    def toMM(self) -> np.array:
        T = np.eye(4)
        T[3, 3] = 1000
        return T @ self.H   

    def calculatePoseError(self, target: np.array) -> tuple:
        # calculate translational error
        t = target[0:3,3] - self.position
        t = t.reshape(3, 1) # col vec

        # calculate rotational error
        E = np.transpose(target[0:3, 0:3]) @ self.H[0:3, 0:3]
        r = np.zeros((3, 1))
        r[0] = E[1, 2] - E[2, 1]
        r[1] = E[2, 0] - E[0, 2]
        r[2] = E[0, 1] - E[1, 0]

        e = np.concatenate((t, 1/2*r))

        return e, np.linalg.norm(e)
        
    @staticmethod
    def eul2rot(eulerAngles: np.array):
        return Transform.rotZ(eulerAngles[2]) @ Transform.rotY(eulerAngles[1]) @ Transform.rotX(eulerAngles[0])

    @staticmethod
    def rot2eul(R: np.array): 
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
        singular = sy < 1e-6
    
        if  not singular :
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else :
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
    
        return np.array([x, y, z]).reshape(3, 1)
    
    @staticmethod
    def rotX(angle: float):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    
    @staticmethod
    def rotY(angle: float):
        return np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    
    @staticmethod
    def rotZ(angle: float):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])