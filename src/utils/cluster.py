import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes

# import bokeh.io as bio
# import plotly.graph_objects as go

import holoviews as hv
hv.extension("bokeh", logo=False)
hv.extension("plotly", logo=False)

def createCentroidDf(centroids: [list], columns: list, predictions: list = None):
  ''' Create data frame containing cluster centroids '''
  # Construct the data frame
  df = pd.DataFrame(centroids, columns=columns)
  
  # Get the number of samples that belong to each cluster
  if (predictions is not None):
    clusters = {}
    for p in predictions:
        if (p not in clusters):
            clusters[p] = 0
        clusters[p] += 1
    clustersSeries = pd.Series(clusters)

    df["Number of Samples"] = clustersSeries

  return df

def calcWcss(centroids: [list], predictions: list, samples: pd.DataFrame):
  ''' Calculate the WCSS value of a clustering algorithm '''
  if (len(predictions) != len(samples)):
    raise ValueError(f"Length of predictions and samples don't match: {len(predictions)} != {len(samples)}")

  # Determine which sample belongs to which cluster
  clusters = groupClusterSamples(predictions)
  
  # Calculate wcss over all clusters
  wcss = 0
  for cluster, sampleIndices in clusters.items():
    centroid = centroids[cluster]
    wcssCluster = 0
    for i in sampleIndices:
      wcssCluster += dissimilarDistance(samples.iloc[i], centroid)
    wcss += wcssCluster
    
  return wcss
  
def dissimilarDistance(sample1, sample2):
  ''' Compute the dissimiliar value between 2 samples '''
  if (len(sample1) != len(sample2)):
    raise ValueError(f"Length of two samples don't match: {len(sample1)} != {len(sample2)}")

  dissimilar = 0
  for value1, value2 in zip(sample1, sample2):
    if (value1 != value2):
      dissimilar += 1
  return dissimilar

def groupClusterSamples(predictions: list) -> dict:
  ''' Group the samples that belong to the same cluster together. 
      Returns: dictionary containing cluster labels as keys and the samples' index that belong to
               their associated clusters as values
  '''
  # Determine which sample belongs to which cluster
  clusters = {}
  for i, p in enumerate(predictions):
    if (p not in clusters):
      clusters[p] = set()
    clusters[p].add(i)
  return clusters

def silhouetteMethod(predictions: list, samples: pd.DataFrame):
  ''' Compute the silhoutte scores for each sample. This computation is
      not optimized hence it is VERY SLOW. As a result, the code was
      translated to C++ for better performance.

      Algorithm: https://en.wikipedia.org/wiki/Silhouette_(clustering)
  '''
  if (len(predictions) != len(samples)):
    raise ValueError(f"Length of predictions and samples don't match: {len(predictions)} != {len(samples)}")

  silhouetteValues = {}
  clusters = groupClusterSamples(predictions)
  for cluster, sampleIndices in clusters.items():

    silhouetteValues[cluster] = []

    for i in sampleIndices:
      print(f"Computing a({i}) in cluster {cluster}")
      sumDistance = 0
      # Compute distance from sample i to all other points in the same cluster
      for j in sampleIndices:
        if (i != j):
          sumDistance += dissimilarDistance(samples.iloc[i], samples.iloc[j])
      a = sumDistance / (len(sampleIndices) - 1)

      # Compute distance from sample i to other samples in other clusters
      bValues = []
      for otherCluster, otherSampleIndices in clusters.items():
        if (cluster != otherCluster):
          print(f"Computing b({i}) in cluster {otherCluster}")
          sumDistance = 0
          for j in otherSampleIndices:
            sumDistance += dissimilarDistance(samples.iloc[i], samples.iloc[j])
          bValue = sumDistance / len(otherSampleIndices)
          bValues.append(bValue)
      b = min(bValues)

      s = ((b - a) / max(a, b)) if len(sampleIndices) > 1 else 0
      silhouetteValues[cluster].append(s)

    # Sort each silhouette value from a cluster from big to small
    silhouetteValues[cluster].sort(reverse=True)

  return silhouetteValues

def readSilFile(silFilePath: str, removeDuplicates: bool =True) -> [float]:
  ''' Read in the silhouette scores computed by C++ program 
      @silFilePath: path to the sil text file
      @removeDuplicates: true if we want to remove duplciated silhouette scores. False otherwise
      Returns: list of silhouette scores from the file
  '''
  silScores = []
  with open(silFilePath, "r") as f:
    for line in f:
      silScores.append(float(line))

  return silScores if (not removeDuplicates) else list(set(silScores))

def averageSilScore(folderPath: str) -> float:
  ''' Calculate the average silhoutte score of a k mode prediction 
      @folderPath: path to where the silhouette files are
      Returns: the average silhouette score
  '''
  filePaths = getSilFilePaths(folderPath)
  sumSilScore = 0.0
  numScores = 0
  for filePath, _ in filePaths:
    silScores = readSilFile(filePath, False)
    sumSilScore += sum(silScores)
    numScores += len(silScores)
  return sumSilScore / numScores

def getSilFilePaths(folderPath: str) -> [str]:
  ''' Get a list of silhouette file paths, generated by the C++ program.
      The file should be in this format: "sil_x.txt" (x = cluster label)
      
      @folderPath: path to where the silhouette files are
      Returns: list of silhouette file paths
  '''
  filePaths = []
  for fileName in os.listdir(folderPath):
    if (fileName.startswith("sil_")):
      filePath = os.path.join(folderPath, fileName)
      fileNameNoExt = os.path.splitext(fileName)[0]
      clusterLabel = int(fileNameNoExt.split('_')[1])

      filePaths.append((filePath, clusterLabel))

  filePaths.sort(key=lambda x: x[1])
  return filePaths  

def plotSilhouetteMethod(folderPath: str, axis: plt.Axes=None) -> plt.Axes:
  ''' Plot a silhouette scores as a horizontal bar chart
      
      @folderPath: path to where the silhouette files are
      @axis: the axis to plot in
      Returns: the plotted axis object
  '''
  filePaths = getSilFilePaths(folderPath)
  avgSilScore = averageSilScore(folderPath)

  data = []
  indices = []
  for filePath, clusterLabel in filePaths:
    # Do not remove duplicates just yet, need to get all the scores
    # Because we want to display the number of samples that belong to a cluster
    silScores = readSilFile(filePath, False)
    indices.append(f"{clusterLabel}_{len(silScores)}")

    # Once we get the total number of samples, then remove duplicate scores
    # so that matplotlib won't be overwhelmed with too many samples
    data.append(set(silScores))

  df = pd.DataFrame(data, index=indices)

  # Plot the silhoutte score and its average score
  ax = df.plot.barh(legend=False, ax=axis)
  avgSilPlot = ax.axvline(avgSilScore, color='k', linestyle='--', 
                          label=f"Average silhoutte = {avgSilScore:.3f}")
  ax.legend(handles=[avgSilPlot])
  
  ax.set_title(f"k = {len(filePaths)}")
  ax.set_xlabel("Silhouette scores")
  ax.set_ylabel("Cluster labels")
  return ax

def plotSilhouetteScores(nClusters: int, df: pd.DataFrame, predictions: [], ax=None, xRange=[-1, 1]):
  '''
      Plot the silhouette scores from a KMeans output.
      Taken from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

      @nClusters: number of clusters
      @df: data frame object
      @predictions: the list of predictions generated by the KMeans algorithm
      @ax: axis object. If None, this function will create a new figure and axis object
      Returns: figure and axis object
  '''
  # Compute the average silhouette score and the silhouette score
  # for each sample 
  silAvg = silhouette_score(df, predictions)
  silValues = silhouette_samples(df, predictions)

  # Create matplotlib figure
  fig = None
  if (ax is None):
    fig, ax = plt.subplots()

  # Set the x and y range of the plot
  # The (n_clusters+1)*10 is for inserting blank space between silhouette
  # plots of individual clusters, to demarcate them clearly.
  ax.set_xlim(xRange)
  ax.set_ylim([0, len(df) + (nClusters + 1) * 10])

  yLower = 10
  for i in range(nClusters):
    clusterSilValues = silValues[predictions == i]
    clusterSilValues.sort()
    
    # Get the cluster size
    clusterSize = clusterSilValues.shape[0]
    yUpper = yLower + clusterSize
    
    # Plot the cluster's silhouette scores
    color = cm.nipy_spectral(float(i) / nClusters)
    ax.fill_betweenx(np.arange(yLower, yUpper),
                     0, clusterSilValues,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    # Annotate the cluster label
    ax.text(-0.05, yLower + 0.5 * clusterSize, str(i))
    yLower = yUpper + 10

  # Plot the average silhouette score  
  ax.axvline(x=silAvg, color="red", linestyle="--", label=f"Avg={silAvg:.3f}")
  ax.legend()

  return fig, ax

def elbowMethod(df, numClusters, clusterAlgType, label="", printInfo=False, categorical=None, **kwargs):
  ''' Run the elbow method for a given cluster k cluster algorithm,
      such as KMeans and KModes. Plot the elbow method graph.

      @df: data frame object
      @numClusters: number of clusters
      @clusterAlgType: the k cluster algorithm object (like KMeans)
      @label: label of the plot
      **kwargs: optional arguments passed in to the @clusterAlgType constructor
  '''
  hv.extension("bokeh")

  wcss = []
  clusterRange = [i for i in range(1, numClusters + 1)]
  for c in clusterRange:
    # Construct the K cluster algorithm object
    kCluster = clusterAlgType(n_clusters=c, **kwargs)

    # Kprototype requires the "categorical" parameter
    if (clusterAlgType is KPrototypes):
      kCluster = kCluster.fit(df, categorical=categorical)
    else:
      kCluster = kCluster.fit(df)

    # Each k cluster algorithm object has different "cost" property
    # So atttemp to find the corresonding cost property
    cost = 0.0
    if (hasattr(kCluster, "cost_")):
      cost = kCluster.cost_
    elif (hasattr(kCluster, "inertia_")):
      cost = kCluster.inertia_
    else:
      raise ValueError(f"Invalid cluster algorithm type {clusterAlgType}")
    
    wcss.append(cost)

    if (printInfo):
      print(f"# clusters: {c} - wcss: {cost}")

  pointsPlot = hv.Points((clusterRange, wcss), label=label).opts(size=10, tools=["hover"])
  linePlot = hv.Curve((clusterRange, wcss))

  return (pointsPlot * linePlot).opts(width=800, height=400, 
          xlabel="Number of cluster", ylabel="WCSS", title="Elbow Method")

def swapTwoColumns(df, c1, c2):
  ''' Swap two columns in a data frame 
      @df: data frame object
      @c1: column 1 name
      @c2: column 2 name
      Returns: a view of the @df with the new column order
  '''
  cols = list(df.columns)
  c1Index, c2Index = cols.index(c1), cols.index(c2)
  cols[c1Index], cols[c2Index] = cols[c2Index], cols[c1Index]
  return df[cols]

def plotKmeansClusters(df: pd.DataFrame, numClusters: int, colors: [], title: str, **kwargs):
  ''' Plot KMeans clusters and their respective centroids. This can only plot 1D, 2D, and 3D data 
      @df: data frame object
      @numClusters: number of clusters
      @colors: list containing color strings to differentiate each cluster
      Returns: a holoview overlay object
  '''
  # Run KMeans
  kmeans = KMeans(n_clusters=numClusters, **kwargs)
  preds = kmeans.fit_predict(df)
  numCols = kmeans.cluster_centers_.shape[1]

  # Display a warning if data has more than 3 dimensions
  if (numCols > 3):
    warnings.warn("[WARNING] This function only supports drawing clusters up to 3 dimensions max.")

  # Construct the prediction data frame to include a color column
  clustersDf = df.copy(True)
  clustersDf["color"] = [colors[p] for p in preds]

  # Construct centroids data frame to include a color column
  # If centroid data dimensions exceed 3, then just use the first 3 dimensions
  columnRange = [i for i in range(numCols)][:3]
  centroids = kmeans.cluster_centers_[:, columnRange]
  centroidColumns = [f"Col_{i}" for i in columnRange]

  # Add a color column to the centroidsDf
  centroidsDf = pd.DataFrame(centroids, columns=centroidColumns)
  centroidsDf["color"] = pd.Series(colors[:len(centroidsDf)])

  # Plot the scatter points in 2D or 3D
  clustersScatter = None
  centroidsScatter = None

  # Scatter plot options
  clustersScatterOpts = {"color": "color", "size": 10}
  centroidsScatterOpts = {"color": "color", "size": 20}

  if (numCols <= 2):
    hv.extension("bokeh")
    # If data is only 1D then use 0 as the y values
    if (numCols == 1):
      # Assign 0 as y column
      clustersDf["Zero_Col"] = 0
      centroidsDf["Zero_Col"] = 0
      
      # Swap the columns so that the zero column is the 2nd column. This
      # allows holoviews to use the 2nd column as the y values
      clustersDf = swapTwoColumns(clustersDf, "color", "Zero_Col")
      centroidsDf = swapTwoColumns(centroidsDf, "color", "Zero_Col")
    
    clustersScatter = hv.Scatter(clustersDf).opts(tools=["hover"], **clustersScatterOpts)
    centroidsScatter = hv.Scatter(centroidsDf).opts(tools=["hover"], marker="triangle", **centroidsScatterOpts)
  else:
    # Use plotly backend. Currently holoviews doesn't support bokeh 3D scatter plot
    hv.extension("plotly")

    clustersScatter = hv.Scatter3D(clustersDf).opts(**clustersScatterOpts)
    centroidsScatter = hv.Scatter3D(centroidsDf).opts(marker="diamond-open", **centroidsScatterOpts)

  # Render both plots on the same figure
  scatterOverlay = (clustersScatter * centroidsScatter).opts(title=title, width=800, height=650)

  return scatterOverlay