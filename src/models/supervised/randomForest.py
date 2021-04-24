import os
import pandas as pd
import numpy as np

import utils.decorators as dc

from models.model import SupervisedModel
from .decisionTree import DecisionTree

class RandomForest(SupervisedModel):
  ''' Random Forest classifer '''

  def __init__(self, maxDepth=3, numberDTrees=3, numberFeaturesToSplit=0):
    ''' Constructor '''
    if (numberDTrees < 0):
      raise ValueError("Number of decision trees must be greater than 0")
    if (numberFeaturesToSplit < 0):
      raise ValueError("Number of features to split must be at least 1")

    super().__init__()
    self._maxDepth = maxDepth
    self._numberFeaturesToSplit = numberFeaturesToSplit

    self._dTrees = []
    self._generateTrees(numberDTrees)

  @property
  def maxDepth(self) -> int:
    ''' '''
    return self._maxDepth

  @maxDepth.setter
  def maxDepth(self, value: int):
    ''' '''
    if (value < 0):
      raise ValueError("Max depth must be at least 0")
    self._maxDepth = value

    # Set max depth to every decision tree
    for dt in self._dTrees:
      dt.maxDepth = self._maxDepth

  @property
  def numberDTrees(self):
    ''' '''
    return len(self._dTrees)

  @numberDTrees.setter
  def numberDTrees(self, value):
    ''' '''
    if (value < 0):
      raise ValueError("Number of decision trees must be greater than 0")
    self._generateTrees(value)

  @property
  def numberFeaturesToSplit(self):
    ''' '''
    return self._numberFeaturesToSplit

  @numberFeaturesToSplit.setter
  def numberFeaturesToSplit(self, value):
    ''' '''
    if (value is not None and value < 0):
      raise ValueError("Number of features to split must be at least 1")
    self._numberFeaturesToSplit = value
    
    # Set number of features to split to every decision tree
    for dt in self._dTrees:
      dt.numberFeaturesToSplit = self._numberFeaturesToSplit

  @property
  def dTrees(self):
    ''' '''
    return self._dTrees

  @property
  def featureImportance(self):
    ''' Random forest calculates feature importance by summing all the importance values of each feature
        across all decision trees. Then divide each sum by the number of decision trees
    '''
    featureImportances = {}

    # Sum up all the importance values of each feature in each decision tree
    for dTree in self._dTrees:
      for _, row in dTree.featureImportance.iterrows():
        feature = row["Feature"]
        importanceValue = row["Value"]
        if (feature not in featureImportances):
          featureImportances[feature] = 0
        featureImportances[feature] += importanceValue

    # Divide each feature importance value by the number of decision tree
    for feature in featureImportances.keys():
      featureImportances[feature] /= self.numberDTrees

    # Convert dict to DataFrame, sorted by "Value" from great to small
    featureImportDf = pd.DataFrame(featureImportances.items(), columns=["Feature", "Value"])
    featureImportDf.sort_values("Value", inplace=True, ascending=False)
    featureImportDf.reset_index(drop=True, inplace=True)
    return featureImportDf

  @dc.elapsedTime
  def train(self, dataFrame, targetSeries, **kwargs):
    ''' '''
    bootStrapSamples = RandomForest.generateBootstrapSamples(self.numberDTrees, dataFrame)
    
    for i, bs in enumerate(bootStrapSamples):
      self._dTrees[i].train(dataFrame.loc[bs], targetSeries.loc[bs], **kwargs) 

  @dc.elapsedTime
  def classify(self, dataFrame, **kwargs):
    ''' '''
    predictions = []
    probabilities = []

    for _, row in dataFrame.iterrows():
      majorityVotes = {}
      
      # Get the prediction from each decision tree
      for dt in self._dTrees:
        prediction, probability = dt.classifyOneSample(row)

        # Initialize the count and average probability to 0
        if (prediction not in majorityVotes):
          majorityVotes[prediction] = {"count": 0, "avgProb": 0}

        majorityVotes[prediction]["count"] += 1
        majorityVotes[prediction]["avgProb"] += probability

      # Compute average probability for each prediction value
      for prediction, value in majorityVotes.items():
        value["avgProb"] /= value["count"]

      # Get the label with the most votes
      # Find the prediction with the most votes and if there is a tie,
      # pick the prediction with the higher average probability
      bestPred, countAndProb = max(majorityVotes.items(), key=lambda kv: (kv[1]["count"], kv[1]["avgProb"]))
      predictions.append(bestPred)
      probabilities.append(countAndProb["avgProb"])

    return pd.DataFrame({"Prediction": predictions, "Probability": probabilities}, index=dataFrame.index)

  def save(self, folderPath, filePrefix, fileFormat):
      ''' Save the graph in each decision tree as individual PDF file '''
      for i, dt in enumerate(self._dTrees):
        filePath = os.path.join(folderPath, f"{filePrefix}_{i + 1}")
        dt.save(filePath, fileFormat)
      return os.path.abspath(folderPath)

  @staticmethod
  def generateBootstrapSamples(k, dataFrame):
    ''' Generate k bootstrap samples
      Returns a a list containing lists of indices.
    '''
    samples = []
    sampleSize = int(len(dataFrame) / k)
    remainer = len(dataFrame) % k

    for _ in range(k-1):
      bootstrapSample = np.random.choice(
        dataFrame.index.values, size=sampleSize, replace=False)
      samples.append(bootstrapSample)

    # Take care of any remainer here
    bootstrapSample = np.random.choice(
      dataFrame.index.values, size=sampleSize + remainer, replace=False)
    samples.append(bootstrapSample)

    return samples

  @staticmethod
  def getCatFeatureValueMappings(dataFrame):
    ''' Get only categorical feature to value mappings '''
    featuresToValues = {}
    for feature in dataFrame.columns.values:
      if (RandomForest.isCategoricalFeature(feature, dataFrame)):
        featuresToValues[feature] = set(dataFrame[feature].unique())
    return featuresToValues

  def _generateTrees(self, numDTrees):
    ''' Generate trees given the random forest properties '''
    # Remove existing trees
    del self._dTrees[:]
    for _ in range(numDTrees):
      dt = DecisionTree(maxDepth=self._maxDepth, numberFeaturesToSplit=self._numberFeaturesToSplit)
      self._dTrees.append(dt)

  def __repr__(self):
    ''' '''
    s = "Model=\"Random Forest\"\nNumber of trees={0}\nMax Depth={1}\nNumber of features to split={2}"
    return s.format( self.numberDTrees, self.maxDepth, self.numberFeaturesToSplit) 

