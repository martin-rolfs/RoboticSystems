import numpy as np

from scipy.interpolate import splprep, splev
from robotic_systems.pose import Transform

class Trajectory:
    def __init__(self, anchorPoints: list, totalTime: float=1.0, startingTime: float=0.0):
        self.anchorPoints = anchorPoints
        self.totalTime = totalTime
        self.startingTime = startingTime

        self.spline = self.constructTrajectory()

    def constructTrajectory(self):
        tck, _ = splprep(self.anchorPoints, s=0, per=1, nest=-1)
        return tck

    def getPoint(self, time: float):
        u = (time - self.startingTime) / self.totalTime
        p = splev(u, self.spline)
        return np.array([p[0], p[1], p[2]])
    
    @staticmethod
    def convertPointForTFC(point: np.array, rcm: np.array, endoscopLength: float) -> Transform:
        diff = rcm - point
        dir = (diff) / np.linalg.norm(diff) 

        tfcPoint = point + dir * endoscopLength
        #print(f"TFC Point: {tfcPoint}")

        xAxis = np.cross(dir, np.array([1, 0, 0]))
        yAxis = np.cross(dir, xAxis)

        tfcTarget = np.array([[xAxis[0], yAxis[0], dir[0], tfcPoint[0]],
                              [xAxis[1], yAxis[1], dir[1], tfcPoint[1]],
                              [xAxis[2], yAxis[2], dir[2], tfcPoint[2]],
                              [       0,        0,      0,           1]])
        
        # rotate final coordinate system 180 degrees around its z axis
        R_y = np.array([[-1.0, 0.0,  0.0, 0.0],
                        [ 0.0, 1.0,  0.0, 0.0],
                        [ 0.0, 0.0, -1.0, 0.0],
                        [ 0.0, 0.0,  0.0, 1.0]])

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