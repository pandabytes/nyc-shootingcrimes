
class Node:
  def __init__(self, entropy, sampleCount):
    ''' '''
    self._entropy = entropy
    self._sampleCount = sampleCount
    self._valuesToNodes = {}

  @property
  def entropy(self):
    return self._entropy

  @property
  def sampleCount(self):
    return self._sampleCount

  def keys(self):
    ''' '''
    return self._valuesToNodes.keys()

  def values(self):
    return self._valuesToNodes.values()

  def items(self):
    ''' '''
    return self._valuesToNodes.items()

  def __getitem__(self, key):
    ''' '''
    return self._valuesToNodes[key]

  def __setitem__(self, key, value):
    ''' '''
    self._valuesToNodes[key] = value

class DecisionNode(Node):
  ''' Class node that contains the split information. It contains the
     the feature is chosen to be split and its value
  '''
  def __init__(self, feature, entropy, sampleCount, numericValue=None):
    if (numericValue is not None and 
        type(numericValue) is not int and 
        type(numericValue) is not float):
      raise ValueError("numericValue must be int or float")
    
    super().__init__(entropy, sampleCount)
    self._feature = feature
    self._numericValue = numericValue

  @property
  def feature(self):
    ''' '''
    return self._feature

  @property
  def numericValue(self):
    ''' '''
    return self._numericValue

  @numericValue.setter
  def numericValue(self, value):
    ''' '''
    if (value is not None and type(value) is not int and type(value) is not float):
      raise ValueError("Value must be int or float")
    self._numericValue = value

  def __str__(self):
    ''' '''
    return f"Feature = {self.feature}\n" + \
           f"Entropy = {self.entropy:.2f}\n" + \
           f"Samples = {self.sampleCount}"

  def __repr__(self):
    ''' '''
    return "Decision Node | Split feature: \"{0}\" | Values: {1}".format(self.feature, list(self.keys()))

class LeafNode(Node):
  ''' Class node that contains the prediction of the sample's label '''
  def __init__(self, prediction, probability, entropy, sampleCount):
    super().__init__(entropy, sampleCount)
    self._prediction = prediction
    self._probability = probability

  @property
  def prediction(self):
    return self._prediction

  @property
  def probability(self):
    return self._probability

  def __str__(self):
    ''' '''
    return f"Prediction = {self.prediction}\n" + \
           f"Probability = {self.probability:.2f}\n" + \
           f"Entropy = {self.entropy:.2f}\n" + \
           f"Samples = {self.sampleCount}"

  def __repr__(self):
    ''' '''
    return "Leaf Node | Prediction: \"{0}\" | Probability: {1}".format(self.prediction, self.probability)