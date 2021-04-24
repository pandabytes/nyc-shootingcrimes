import os
import math
import numpy as np
import pandas as pd
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.decomposition import PCA

import holoviews as hv
hv.extension("bokeh", logo=False)


def filterRowsByValue(data: pd.DataFrame, undesiredValue: object, maxOccurence: int) -> pd.DataFrame:
  ''' Filter the rows that contain the undesiredValue IF the number of occurences
      of such value is greater than maxOccurence.
      @data: the data frame
      @undesiredValue: the value that we want to filter
      @maxOccurence: the max number of occurence of the undesiredValue should appear in a row
      Returns: a filtered view of the original data frame
  '''
  removeRowIndices = []
  for i, row in data.iterrows():
    undesiredValueCount = 0
    for col in data.columns.values:
      if (row[col] == undesiredValue):
        undesiredValueCount += 1

    if (undesiredValueCount > maxOccurence):
      removeRowIndices.append(i)

  return data[~data.index.isin(removeRowIndices)]

def getSampleCountByValueOccurence(data: pd.DataFrame, value: object) -> dict:
  ''' Get the occurences of a value and its associated number of samples that have such occurences
      @data: the data frame
      @value: the value we use to find its number of occurences and its associated number of samples that
              contain such occurences
      Returns: dictonary where each key is the number of occurence and value is the number of samples
  '''
  d = {}
  for _, row in data.iterrows():
    valueCount = 0
    for col in data.columns.values:
      # Count how many times value appears
      if (row[col] == value):
        valueCount += 1

    # Store in the dictionary
    if (valueCount not in d):
      d[valueCount] = 0
    d[valueCount] += 1

  return d

def getCorrelationDf(df: pd.DataFrame, correlationMethod: str):
  ''' Get the correlation data frame using the given @df and @correlationMethod 
      @df: data frame object
      @correlationMethod: a correlation function. This must be either cramersV, theilU, correlationRatio, or correlation
      Returns: data frame object with correlation values
  '''
  funcNames = [theilU.__name__, cramersV.__name__, correlationRatio.__name__, "correlation"]
  if (correlationMethod not in funcNames):
    raise ValueError(f"Correlation method must be one of these: {funcNames}")

  correlationDf = None

  # Determine cat and cont features
  numFeatures = df.select_dtypes(include=[np.number, "bool"]).columns.values
  catFeatures = df.select_dtypes(include=["object", "bool", "category"]).columns.values

  if (correlationMethod == "correlation"):
    correlationDf = df[numFeatures].corr()
  elif (correlationMethod == correlationRatio.__name__):
    correlationDf = pd.DataFrame(columns=catFeatures, index=numFeatures, dtype=np.number)

    for cat in catFeatures:
      for cont in numFeatures:
        correlation = correlationRatio(df[cat], df[cont])
        correlationDf.loc[cont, cat] = correlation
  else:
    correlationFunc = theilU if (correlationMethod == theilU.__name__) else cramersV
    correlationDf = pd.DataFrame(columns=catFeatures, index=catFeatures, dtype=np.number)
    for c1 in catFeatures:
      for c2 in catFeatures:
        correlation = correlationFunc(df[c1], df[c2])
        correlationDf.loc[c1, c2] = correlation

  return correlationDf

def __conditionalEntropy(x, y):
  '''
      https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9 
  '''
  # Entropy of x given y
  y_counter = Counter(y)
  xy_counter = Counter(list(zip(x,y)))
  total_occurrences = sum(y_counter.values())
  entropy = 0
  for xy in xy_counter.keys():
    p_xy = xy_counter[xy] / total_occurrences
    p_y = y_counter[xy[1]] / total_occurrences
    entropy += p_xy * math.log(p_y/p_xy)
  return entropy

def theilU(x, y):
  '''
      https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9 
  '''
  s_xy = __conditionalEntropy(x,y)
  x_counter = Counter(x)
  total_occurrences = sum(x_counter.values())
  p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
  s_x = ss.entropy(p_x)
  
  if s_x == 0:
    return 1
  
  return (s_x - s_xy) / s_x

def cramersV(x, y):
  '''
      https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
  '''
  confusion_matrix = pd.crosstab(x,y)
  chi2 = ss.chi2_contingency(confusion_matrix)[0]
  n = confusion_matrix.sum().sum()
  phi2 = chi2/n
  r, k = confusion_matrix.shape
  phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
  rcorr = r-((r-1)**2)/(n-1)
  kcorr = k-((k-1)**2)/(n-1)
  return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def correlationRatio(categoricalValues, continousValues):
  '''
  https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
  '''
  fcat, _ = pd.factorize(categoricalValues)
  cat_num = np.max(fcat)+1
  y_avg_array = np.zeros(cat_num)
  n_array = np.zeros(cat_num)

  for i in range(0, cat_num):
      cat_measures = continousValues[np.argwhere(fcat == i).flatten()]
      n_array[i] = len(cat_measures)
      y_avg_array[i] = np.average(cat_measures)

  y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
  numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
  denominator = np.sum(np.power(np.subtract(continousValues,y_total_avg),2))

  return np.sqrt(numerator/denominator)

def printCollectionInColumns(collection, numCol=3):
  ''' Pretty print a collection in n columns '''
  remainer = int(len(collection) % numCol)
  rowCount = int(len(collection) / numCol) + remainer
  longestStr = max(map(lambda x: len(x), collection))
  
  startIndex = 0
  endIndex = startIndex + numCol
  
  for _ in range(rowCount):
      row = collection[startIndex:endIndex]
      strFormat = "".join([f"{{{j}:<{longestStr + 2}}}" 
                            for j in range(len(row))])
      rowStr = strFormat.format(*row)
      print(rowStr)
      
      startIndex = endIndex
      endIndex = startIndex + numCol

def computeError(predictions, actuals):
  ''' Compute the % of misclassification of the prediction '''
  if (len(predictions) != len(actuals)):
    raise ValueError(f"Number of predictions and actuals must match: {len(predictions)} != {len(actuals)}")
  
  misClassified = 0
  for i, j in zip(predictions, actuals):
    if (i != j):
      misClassified += 1
  return misClassified / len(actuals)

def groupByAndUnstack(column: str, df: pd.DataFrame) -> pd.DataFrame:
  ''' Group the given @df by the @column. Then perform an unstack operation
      to create new columns filled with column value counts of each column in @df.
      @column: column name
      @df: data frame object
      Returns: a data fram object resulted from concatenation of multiple unstacked
               data frame object.
  '''
  unstackDfs = []

  groups = df.groupby(column)
  columns = [c for c in groups.obj.columns if (c != column)]
  for col in columns:
    unstackDf = groups[col].value_counts().unstack().fillna(0)
    
    # Append the original column name to each new column
    unstackDf.columns = [f"{col}_{c}" for c in unstackDf.columns]
    
    unstackDfs.append(unstackDf)

  return pd.concat(unstackDfs, axis=1, sort=False)

def plotPcaVarianceRatio(numComponents: int, df: pd.DataFrame):
  ''' Plot the PCA variance ratio 
      @numComponents: number of principle components
      @df: the data frame object
      Returns: an overlay of bokeh plots
  '''
  pca = PCA(n_components=numComponents)
  pca.fit(df)
  cumsum = np.cumsum(pca.explained_variance_ratio_)
  
  pointsPlot = hv.Points(cumsum).opts(size=10, tools=["hover"])
  linePlot = hv.Curve(cumsum)

  return (pointsPlot * linePlot).opts(width=800, height=400, 
          xlabel="Number of components", ylabel="Variance Ratio", title="PCA Variance Ratio")

def getPcByVariancePercentInterval(startInterval: float, endInterval: float, 
                                   numComponents: int, df: pd.DataFrame) -> [(int, float)]:
  ''' Get the principle components such that their variance are within the specified interval. 

      @startInterval: the start of the interval. Value between [0, 1]
      @endInterval: the end of the interval. Value between [0, 1]
      @@numComponents: number of principle components
      @df: the data frame object
      Returns: a list of the principle components and their variance ratios
  '''
  if (startInterval < 0.0 or startInterval > 1.0):
    raise ValueError("Start interval value must be between [0, 1]")
  if (endInterval < 0.0 or endInterval > 1.0):
    raise ValueError("End interval value must be between [0, 1]")

  pca = PCA(n_components=numComponents)
  pca.fit(df)
  cumsum = np.cumsum(pca.explained_variance_ratio_)
  
  pcs = []
  for i, sumRatio in enumerate(cumsum):
    if (sumRatio >= startInterval and sumRatio <= endInterval):
      pcs.append((i+1, sumRatio))

  return pcs
