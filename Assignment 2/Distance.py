import numpy as np
import math

class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        return 1-((np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)))
    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        return np.sum(((x-y)**p)**(1/p))
    @staticmethod
    def calculateMahalanobisDistance(x,y, S_minus_1):
        return np.sqrt(np.dot(np.dot((x-y).T, S_minus_1),(x-y)))
