import numpy as np

from scipy.interpolate import splprep, splev
from robotic_systems.pose import Transform

class Trajectory:
    def __init__(self, anchorPoints: list, totalTime: float=1.0, startingTime: float=0.0):
        self.anchorPoints = anchorPoints
        self.totalTime = totalTime
        self.startingTime = startingTime
        self.anchorValues = []
        self.dimension = 3

        self.spline = self.constructTrajectory()

    def constructTrajectory(self):
        self.dimension = np.array(self.anchorPoints).shape[0]
        tck, self.anchorValues = splprep(self.anchorPoints, s=0, per=1, nest=-1)
        return tck

    def getPoint(self, time: float):
        u = (time - self.startingTime) / self.totalTime
        p = splev(u, self.spline)
        return np.array(p)
    
    def checkAnchorValues(self, index: int, time: float) -> int:
        """Checks whether point at specified time lies before or after a anchor point which was used for construction.

        Args:
            index (int): The index of the anchor point.
            time (float): The time of the point on the trajectory.

        Returns:
            int: Returns -1 if point comes before anchor point, 1 if it comes after and 0 if it lies directly on anchor point.
        """
        u = (time - self.startingTime) / self.totalTime

        if self.anchorValues[index] > u:
            return -1
        elif self.anchorValues[index] == u:
            return 0
        else:
            return 1
    
    @staticmethod
    def convertPointForTFC(point: np.array, rcm: np.array, tfc_to_rcm_tanslation: np.array, rotate: bool=True, referenceVector: np.array=None, moveInRCM: bool=False) -> Transform:
        diff = rcm - point
        dir = (diff) / np.linalg.norm(diff)            

        tfcPoint = point + ((dir * (np.linalg.norm(diff) + np.linalg.norm(tfc_to_rcm_tanslation))) if not moveInRCM else diff)

        # create unit vector
        if type(referenceVector) == type(None):
            e = np.zeros((3,))
            e[np.argmin(dir)] = 1
        else:
            e = referenceVector

        xAxis = np.cross(dir, e)
        yAxis = np.cross(dir, xAxis)

        tfcTarget = np.array([[xAxis[0], yAxis[0], dir[0], tfcPoint[0]],
                              [xAxis[1], yAxis[1], dir[1], tfcPoint[1]],
                              [xAxis[2], yAxis[2], dir[2], tfcPoint[2]],
                              [       0,        0,      0,           1]])
        
        # rotate final coordinate system 180 degrees around its z axis
        if rotate:
            R_y = np.array([[-1.0, 0.0,  0.0, 0.0],
                            [ 0.0, 1.0,  0.0, 0.0],
                            [ 0.0, 0.0, -1.0, 0.0],
                            [ 0.0, 0.0,  0.0, 1.0]])
        else:
            R_y = np.eye(4)

        return Transform(None, None, tfcTarget @ R_y)

    @staticmethod
    def convertPointList(points: list) -> list:
        """Converts list of np.array to list of three np.array holding each the seperate coordinates.

        Args:
            points (list): List of np.array

        Returns:
            list: List of coordinates compatible with Trajectory constructor.
        """
        return [np.array(p) for p in np.array(points).T]