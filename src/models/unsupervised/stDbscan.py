import pandas as pd
import numpy as np
import os

from typing import Union, Callable
from sklearn.neighbors import NearestNeighbors
from models.model import UnsupervisedModel

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/_dbscan_inner.pyx
from sklearn.cluster._dbscan_inner import dbscan_inner

class StDbscan(UnsupervisedModel):
  ''' '''
  def __init__(self, eps1: float = 0.5, eps2: float = 10, 
               minSamples: int = 5, 
               metric1: Union[str, Callable] = "haversine", metric1Params: dict = None,
               metric2: Union[str, Callable] = "euclidean", metric2Params: dict = None):
    ''' '''
    self.eps1 = eps1
    self.eps2 = eps2
    self.minSamples = minSamples

    self.metric1 = metric1
    self.metric1Params = metric1Params

    self.metric2 = metric2
    self.metric2Params = metric2Params

    self.labels = None
    self.corePoints = None
    self.neighborhoods = None

  def fit(self, dataFrame: pd.DataFrame, spatialFeatures: list, temporalFeatures: list):
    ''' '''
    if (len(spatialFeatures) <= 0 or len(temporalFeatures) <= 0):
      raise ValueError("spatialFeatures and temporalFeatures must be lists with length greater than 0")

    nnEps1 = NearestNeighbors(radius=self.eps1, algorithm="auto",
                              leaf_size=30, metric=self.metric1,
                              metric_params=self.metric1Params, p=None, n_jobs=None)

    nnEps2 = NearestNeighbors(radius=self.eps2, algorithm="auto",
                              leaf_size=30, metric=self.metric2,
                              metric_params=self.metric2Params, p=None, n_jobs=None)
    
    spatialDf = dataFrame[spatialFeatures]
    temporalDf = dataFrame[temporalFeatures]
    nnEps1.fit(spatialDf)
    nnEps2.fit(temporalDf)

    eps1Neighborhoods = nnEps1.radius_neighbors(spatialDf, return_distance=False)
    eps2Neighborhoods = nnEps2.radius_neighbors(temporalDf, return_distance=False)

    # Intersection of 2 neighborhoods
    neighborhoods = []
    for eps1Neighbors, eps2Neighbors in zip(eps1Neighborhoods, eps2Neighborhoods):
      intersection = np.intersect1d(eps1Neighbors, eps2Neighbors)
      neighborhoods.append(intersection)
    neighborhoods = np.array(neighborhoods, dtype=object)

    # All samples are noise in the beginning
    labels = np.full(dataFrame.shape[0], -1, dtype=np.intp)

    # 
    neighborsCounts = np.array([len(neighborhood) for neighborhood in neighborhoods])
    corePoints = np.asarray(neighborsCounts >= self.minSamples, dtype=np.uint8)

    dbscan_inner(corePoints, neighborhoods, labels)
    self.labels = labels
    self.neighborhoods = neighborhoods
    self.corePoints = corePoints

  def fitPredict(self, dataFrame: pd.DataFrame, spatialFeatures: list, temporalFeatures: list) -> [int]:
    ''' '''
    self.fit(dataFrame, spatialFeatures, temporalFeatures)
    return self.labels
    





