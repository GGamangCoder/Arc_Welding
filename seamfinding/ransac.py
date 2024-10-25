from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import preprocessing
import random

# RANSAC 3D Class

class RANSAC_3D_Class:
    def __init__(self, iter_max, offset):
        self.iter_max = iter_max
        self.offset = offset
        
        self.inliner_max = 0
        self.ransac_plane = list()
        
        self.x = 0
        self.y = 0
        self.z = 0
        
        self.x_select = 0
        self.y_select = 0
        self.z_select = 0
        
    def setData(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def getRANSAC(self):
        return self.ransac_plane
    
    # 1. Select the points randomly
    def selectPoints(self):
        self.x_select = random.sample(self.x.tolist(), 3)
        self.y_select = random.sample(self.y.tolist(), 3)
        self.z_select = random.sample(self.z.tolist(), 3)
    
    def makeHypothesisPlane(self):
        # 2. Plane equation
        point1 = np.array([self.x_select[0], self.y_select[0], self.z_select[0]])
        point2 = np.array([self.x_select[1], self.y_select[1], self.z_select[1]])
        point3 = np.array([self.x_select[2], self.y_select[2], self.z_select[2]])

        v12 = point2 - point1
        v13 = point3 - point1

        n = np.cross(v12, v13)
        d = -np.inner(n, point1)

        # Plane equation
        X = np.arange(-300, 300, 10)
        Y = np.arange(-300, 300, 10)
        X, Y = np.meshgrid(X, Y)
        Z = (-n[0]/n[2] * X) + (-n[1]/n[2] * Y) - d/n[2]

        # 3. Calculate the number of the inliner points
        inliner_cnt = 0

        for i in range(len(Z)):

            check_z_upper = (-n[0]/n[2] * X[i]) + (-n[1]/n[2] * Y[i]) - d/n[2] + self.offset
            check_z_lower = (-n[0]/n[2] * X[i]) + (-n[1]/n[2] * Y[i]) - d/n[2] - self.offset

            if((Z[i] < check_z_upper).any() and (Z[i] > check_z_lower).any()):
                inliner_cnt = inliner_cnt + 1

        # 4. Find the maximum inliner points
        if(self.inliner_max < inliner_cnt):
            self.inliner_max = inliner_cnt
            self.ransac_plane = Z
        
    def processRANSAC(self):
        for iteration in range(self.iter_max):
            self.selectPoints()
            self.makeHypothesisPlane()
