import math
import os
import random
import sys

import numpy as np
import pandas as pd
import models.utils.decorators as dc

from graphviz import Digraph
from .treeNode import DecisionNode, LeafNode
from models.model import SupervisedModel

class DecisionTree(SupervisedModel):
  ''' '''
  ContinuousKeyFormat = "{0}{1:.3f}"

  def __init__(self, maxDepth=3, numberFeaturesToSplit=0):
    ''' Constructor '''
    if (maxDepth < 0):
      raise ValueError("Max depth must be at least 0")
    if (numberFeaturesToSplit < 0):
      raise ValueError("Number of features to split must be at least 1")

    super().__init__()
    self._maxDepth = maxDepth
    self._numberFeaturesToSplit = numberFeaturesToSplit
    self._rootNode = None
    self._diGraph = Digraph("G", format="png")
    self._dataFrame = None
    self._targetSeries = None

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

  @property
  def depth(self) -> int:
    ''' Get the depth of the tree '''
    return self._countTreeDepth(self._rootNode)

  @property
  def numLeafNodes(self) -> int:
    ''' Get the number of leaf nodes in the tree '''
    return self._countLeafNodes(self._rootNode)

  @property
  def treeStructure(self):
    ''' '''
    return self._treeStructure(self._rootNode)

  @property
  def graph(self) -> Digraph:
    ''' Get the graph object representing this decision tree
        @return: a Digraph object from graphviz library
    '''
    self._diGraph.clear()
    self._generateGraph(self._rootNode)
    return self._diGraph

  @property
  def featureImportance(self) -> dict:
    ''' Return a dictionary of features and their associated importance values 
    '''
    featureImportances = {}
    self._calcFeatureImportance(self._rootNode, self._rootNode.sampleCount, featureImportances)
    
    # Convert dict to DataFrame, sorted by "Value" from great to small
    featureImportDf = pd.DataFrame(featureImportances.items(), columns=["Feature", "Value"])
    featureImportDf.sort_values("Value", inplace=True, ascending=False)
    featureImportDf.reset_index(drop=True, inplace=True)
    return featureImportDf

  @dc.elapsedTime
  def train(self, dataFrame, targetSeries, **kwargs):
    ''' '''
    self._dataFrame = dataFrame
    self._targetSeries = targetSeries

    features = [f for f in self._dataFrame.columns]
    self._rootNode = self.buildTree(features, self._dataFrame.index.values, 0)
    
    self._dataFrame = None
    self._targetSeries = None

  def classify(self, dataFrame):
    ''' '''
    predictions = []
    probabilities = []

    for _, row in dataFrame.iterrows():
      prediction, probability = self.classifyOneSample(row)
      predictions.append(prediction)
      probabilities.append(probability)

    return pd.DataFrame({"Prediction": predictions, "Probability": probabilities}, index=dataFrame.index)

  def classifyOneSample(self, sample):
    ''' Wrapper method for _classifyOneSample(). This abstracts away the root node from being the required argument. '''
    return self._classifyOneSample(sample, self._rootNode)

  def buildTree(self, features, indices, depth):
    ''' '''
    subDataFrame = self._dataFrame.loc[indices]
    subTargetSeries = self._targetSeries.loc[indices]

    if (depth >= self._maxDepth):
      return self._constructLeafNode(subTargetSeries)
      
    # Consider a subset of features to split
    subsetFeatures = self.getRandomFeatures(features, self.numberFeaturesToSplit)
    bestFeature, value, infoGain = self.findBestFeature(subsetFeatures, indices)
    if (infoGain == 0 or bestFeature is None):
      return self._constructLeafNode(subTargetSeries)

    # Create decision node
    entropy = self.getEntropy(subTargetSeries)
    parentNode = DecisionNode(bestFeature, entropy, len(subDataFrame))
    partitions = None
    
    # Partition data depending on its feature type
    if (self.isNumericFeature(bestFeature, subDataFrame)):
      partitions = self.partitionContinuous(bestFeature, value, subDataFrame)
      parentNode.numericValue = value # Store the splitting value for numeric feature
    else:
      partitions = self.partitionCategorical(bestFeature, subDataFrame)

    # Remove the feature that we used to split
    newFeatures = [f for f in features if f != bestFeature]

    for splitValue, childIndices in partitions.items():
      childNode = self.buildTree(newFeatures, childIndices, depth + 1)
      parentNode[splitValue] = childNode

    return parentNode

  def findBestFeature(self, features, indices):
    '''  '''
    bestInfoGain = -sys.maxsize
    bestFeature = None
    bestFeatureValue = None

    parentDataFrame = self._dataFrame.loc[indices]
    parentTargetSeries = self._targetSeries.loc[indices]
    parentEntropy = self.getEntropy(self._targetSeries.loc[indices])
    parentCount = len(parentDataFrame)

    for feature in features:
      # For continuous features, we use different quantile values
      # to determine the best split value
      if (SupervisedModel.isNumericFeature(feature, parentDataFrame)):
        quantiles = [0.2, 0.4, 0.6, 0.8]
        quantileValues = parentDataFrame[feature].quantile(quantiles)

        for q in quantileValues:
          childrenIndices = self.partitionContinuous(feature, q, parentDataFrame).values()
          infoGain = self.informationGain(
            parentEntropy, parentCount, parentTargetSeries, childrenIndices, )

          if (bestInfoGain < infoGain):
            bestInfoGain = infoGain
            bestFeature = feature
            bestFeatureValue = q
      else:
        childrenIndices = self.partitionCategorical(feature, parentDataFrame).values()
        infoGain = self.informationGain(
          parentEntropy, parentCount, parentTargetSeries, childrenIndices)

        if (bestInfoGain < infoGain):
          bestInfoGain = infoGain
          bestFeature = feature
          bestFeatureValue = None

    return bestFeature, bestFeatureValue, bestInfoGain

  def printTreeStructure(self):
    ''' '''
    def _printTreeStructure(tabSpace, value):
      for k, v in value.items():
        print(tabSpace, k, sep="")
        _printTreeStructure(tabSpace + tabSpace, v)

    _printTreeStructure(" ", self.treeStructure)

  def save(self, filePath, fileFormat):
    ''' Save the graph in a file in the format specified by the parameter fileFormat (pdf, png, etc) '''
    # graphviz includes the format extension after it saves the graph,
    # we need to remove the file extension from filePath
    fileName = os.path.splitext(filePath)[0]
    savedFilePath = self.graph.render(fileName, format=fileFormat, cleanup=True)
    return os.path.abspath(savedFilePath)

  @staticmethod
  def _constructContinuousKey(greatOrLessSign, value):
    ''' '''
    if (greatOrLessSign != "<" and greatOrLessSign != ">" and
        greatOrLessSign != "<=" and greatOrLessSign != ">="):
      raise ValueError("Incorrect inequality sign. Must be either: <, >, <=, or >=.")

    return DecisionTree.ContinuousKeyFormat.format(greatOrLessSign, value)

  @staticmethod
  def partitionCategorical(feature, dataFrame):
    ''' '''
    return dataFrame.groupby(feature).groups

  @staticmethod
  def partitionContinuous(feature, value, dataFrame):
    ''' '''
    leftIndices = dataFrame[dataFrame[feature] < value].index.values
    rightIndices = dataFrame[dataFrame[feature] >= value].index.values
  
    partitions = {}
    if (len(leftIndices) > 0):
      leftKey = DecisionTree._constructContinuousKey("<", value)
      partitions[leftKey] = leftIndices
    
    if (len(rightIndices) > 0):
      rightKey = DecisionTree._constructContinuousKey(">=", value)
      partitions[rightKey] = rightIndices
    
    return partitions

  @staticmethod
  def informationGain(parentEntropy, parentCount, targetSeries, childrenIndices):
    ''' '''
    childrenEntropy = 0
    for childIndices in childrenIndices:
      probability = len(childIndices) / parentCount
      childEntropy = DecisionTree.getEntropy(targetSeries[childIndices])
      childrenEntropy += (probability * childEntropy)
    return parentEntropy - childrenEntropy

  @staticmethod
  def getEntropy(series):
    ''' '''
    seriesCount = len(series)
    if (seriesCount == 0):
      return 0

    resultEntropy = 0
    for _, count in series.value_counts().items():
      probability = count / float(seriesCount)
      if (probability > 0):
        resultEntropy += probability * math.log(probability, 2)

    # Add 0 because we may get "-0" entropy. This is for display 0 without the - sign
    return -resultEntropy + 0

  @staticmethod
  def getGiniImpurity(series):
    ''' @TODO '''
    seriesCount = float(len(series))
    if (seriesCount == 0):
      return 0

    impurity = 0
    for _, count in series.value_counts().items():
      probability = count / seriesCount
      impurity += probability * (1 - probability)

    return impurity

  @staticmethod
  def  _constructLeafNode(series):
    ''' '''
    predictionCount = series.value_counts()
    bestLabel, bestLabelCount = max(predictionCount.items(), key=lambda x: x[1])
    bestProb = float(bestLabelCount) / sum(predictionCount)
    entropy = DecisionTree.getEntropy(series)
    return LeafNode(bestLabel, bestProb, entropy, len(series))

  def _generateGraph(self, node, nodeId=0):
    ''' Generate the decision tree graph. 
    
      @node: the root node of the decision tree
      @nodeId: use to help assign unique id to each node.
      @return: the most current node ID. This is only used to keep track of the most current node ID.
    '''
    if (node is None):
      return nodeId

    # If the root node is a leaf node
    if (type(node) is LeafNode):
      self._addNode(LeafNode, nodeId, str(node))
      return nodeId
    
    # Decision Node starts here
    self._addNode(DecisionNode, nodeId, str(node))
    childNodeId = nodeId + 1

    for featureValue, childNode in node.items():
      if (type(childNode) is LeafNode):
        self._addNode(LeafNode, childNodeId, str(childNode))
        self._addEdge(nodeId, childNodeId, featureValue)
        childNodeId += 1
      else:
        # Node ID will be updated through recursion so we need to save it simply by returning the most current node ID
        currentChildNodeId = self._generateGraph(childNode, childNodeId)
        self._addEdge(nodeId, childNodeId, featureValue)
        childNodeId = currentChildNodeId

    return childNodeId
        
  def _addEdge(self, fromId, toId, nodeLabel):
    ''' '''
    self._diGraph.edge(str(fromId), str(toId), label=nodeLabel)

  def _addNode(self, nodeType, nodeId, nodeLabel):
    ''' '''
    if (nodeType is LeafNode):
      self._diGraph.node(str(nodeId), nodeLabel, color="red")
    elif (nodeType is DecisionNode):
      self._diGraph.node(str(nodeId), nodeLabel)
    else:
      raise ValueError("Invalid node type \"{0}\"".format(nodeType))

  @staticmethod
  def getRandomFeatures(features, numberFeatures):
    ''' Get m random features that we consider to split at each level of the tree 

      @features: list of features
      @numberFeatures: number of features that we want to use. 0 means use all features
    '''
    if (numberFeatures < 0):
      raise ValueError("numberFeatures must be greather or equal to 0")

    randomFeatures = None

    # If 0 then return the same features list OR
    # If the size of the features is <= than the number of features that
    # we want to split, then we simply return all features
    if (numberFeatures == 0 or len(features) <= numberFeatures):
      randomFeatures = features
    else:
      randomFeatures = np.random.choice(features, size=numberFeatures, replace=False)

    return randomFeatures

  def _classifyOneSample(self, sample, node):
    ''' Classify only 1 sample of data. Use recursion. '''
    if (type(node) is LeafNode):
      return node.prediction, node.probability
    else:
      sampleValue = sample[node.feature]

      # Check if this split is from continous feature
      if (node.numericValue is not None):
        # Numeric feature only has 2 branches
        key = None
        if (sampleValue < node.numericValue): 
          key = self._constructContinuousKey("<", node.numericValue)
        else:
          key = self._constructContinuousKey(">=", node.numericValue)

        return self._classifyOneSample(sample, node[key])

      else: # Categorical feature
        if (sampleValue not in node.keys()):
          # Temporarily comment this message print because it will be printed a lot
          # msg = f"Encounter unknown value {sampleValue} of feature {node.feature}. " + \
          #      "Reason is a training node does NOT contain this value. " + \
          #      f"The node contains these values for feature {node.feature}: {list(node.keys())}"
          # print(msg)
          
          # Since the node doesn't contain the unknown value, 
          # we take the the most voted prediction from the 
          # sibling nodes and average the probabilities
          majorityVotes = {}

          # Get the prediction and probability from the "siblings'" predictions
          for v in node.values():
            prediction, probability = self._classifyOneSample(sample, v)
            
            if (prediction not in majorityVotes):
              majorityVotes[prediction] = {"count": 0, "avgProb": 0}
            
            # Update a prediction count so we know which one has the highest count
            majorityVotes[prediction]["count"] += 1
            majorityVotes[prediction]["avgProb"] += probability
          
          # Average probability for each prediction
          for prediction, value in majorityVotes.items():
            value["avgProb"] /= value["count"]
          
          # Get the prediction that has the highest count. If there are multiple n highest counts,
          # then get the prediction that has a higher average probability. Else, get whichever one
          bestLabel, countAndProb = max(majorityVotes.items(), key=lambda kv: (kv[1]["count"], kv[1]["avgProb"]))
          return bestLabel, countAndProb["avgProb"]

        return self._classifyOneSample(sample, node[sampleValue])

  def _treeStructure(self, node):
    ''' '''
    if (node is None):
      return {}

    # Check if the current node only contains leaf nodes
    hasOnlyLeafNodes = len([v for v in node.values() if type(v) is DecisionNode]) == 0
    if (hasOnlyLeafNodes): 
      return {node.feature: {}}
    else:
      structure = {node.feature: {}}
      for i, (featureValue, childNode) in enumerate(node.items()):
        if (type(childNode) is not LeafNode):
          childNodeFeature, childNodeChildren = list(self._treeStructure(childNode).items())[0]
          key = f"{featureValue} --> {childNodeFeature}"
          structure[node.feature][key] = childNodeChildren
        else:
          key = f"Leaf Node {i}"
          structure[node.feature][key] = {}

      return structure
      
  def _countLeafNodes(self, node) -> int:
    ''' Helper function for counting leaf nodes 
    
      @node: root node of the decision tree node
      @return: number of leaf nodes
    '''
    if (node is None):
      return 0
    elif (type(node) is LeafNode):
      return 1
    else:
      count = 0
      for featureValue in node.keys():
        count += self._countLeafNodes(node[featureValue])
      return count
  
  def _countTreeDepth(self, node) -> int:
    ''' Helper function for counting the tree depth 
    
      @node: root node of the decision tree
      @return: the depth of the decision tree
    '''
    if (node is None) or (type(node) is LeafNode) or (len(node.keys()) == 0):
      return 0
    else:
      deepestDepth = 0
      for featureValue in node.keys():
        depth = 1 + self._countTreeDepth(node[featureValue])
        if (deepestDepth < depth):
          deepestDepth = depth
      return deepestDepth

  @staticmethod
  def _calcFeatureImportance(node, totalSampleCount, featureImportances):
    ''' Compute the feature importance of the given node and its descendant nodes recursively 

        Feature importance is calculated by 
          (currentNode.sampleCount / totalSampleCount) 
    '''
    if (isinstance(node, DecisionNode)):
      # Compute the importance value for the current node
      childrenImpurity = 0
      for childNode in node.values():
        if (isinstance(childNode, DecisionNode)):
          childrenImpurity += childNode.entropy * (childNode.sampleCount / node.sampleCount)
      importance = (node.sampleCount / totalSampleCount) * (node.entropy - childrenImpurity)

     # Store it in the dictionary
      if (node.feature not in featureImportances):
        featureImportances[node.feature] = 0
      featureImportances[node.feature] += importance

      # Recursively compute the descendant nodes' importance value
      for childNode in node.values():
        DecisionTree._calcFeatureImportance(childNode, totalSampleCount, featureImportances)

  def __repr__(self):
    ''' '''
    s = "Decision Tree | Depth={0} | Number of Leaf Nodes={1} | Number of features to split={2}"
    return s.format(self.depth, self.numLeafNodes, self.numberFeaturesToSplit) 
