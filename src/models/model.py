import numpy as np
import pandas as pd

class SupervisedModel:
  ''' '''
  def __init__(self):
    ''' '''
    self._targetSeries = None

  def train(self, dataFrame, **kwargs):
    ''' '''
    raise NotImplementedError("Method \"train\" is not implemented")

  def classify(self, dataFrame):
    ''' '''
    raise NotImplementedError("Method \"classify\" is not implemented")

  @staticmethod
  def isNumericFeature(feature, dataFrame):
    ''' '''
    numericDf = dataFrame.select_dtypes(include=[np.number])
    return feature in numericDf.columns.values

  @staticmethod
  def isCategoricalFeature(feature, dataFrame):
    ''' '''
    categoricalDf = dataFrame.select_dtypes(include=["object", "bool", "category"])
    return feature in categoricalDf.columns.values

class UnsupervisedModel:
  ''' '''
  def __init__(self):
    pass

  def fit(self, dataFrame: pd.DataFrame, **kwargs):
    ''' '''
    raise NotImplementedError("Method \"fit\" is not implemented")

  def fitPredict(self, dataFrame: pd.DataFrame, **kwargs):
    ''' '''
    raise NotImplementedError("Method \"fitPredict\" is not implemented")