import os
import pandas as pd

from itertools import product

import holoviews as hv
hv.extension("bokeh", logo=False)

def freqHeatMap(data: pd.DataFrame, column1: str, column2: str, allColumnValues1: [] = None, 
                allColumnValues2: [] = None, **kwargs):
  ''' Create a frequency heatmap of 2 columns of a data frame
      @data: data frame object
      @column1: name of the first column
      @column2: name of the second column
      @allColumnValues1: list of all categorical values of @column1
      @allColumnValues2: list of all categorical values of @column2
      @kwargs: optional parameters for heatmap options
      Returns: heatmap object
  '''
  if (column1 == column2):
    raise ValueError("Column 1 and Column 2 must be different.")

  # Use groupby to get frequency
  filteredData = data.groupby([column1, column2]).size().reset_index(name="Frequency")

  # If specified, populate the data frame with combinations of values that don't currently
  # exist in the data frame. So that all combinations are stored in the data frame and
  # holoviews heatmap can correctly sort the axis labels
  if (allColumnValues1 is not None and allColumnValues2 is not None):
    allCombs = {(i, j) for i, j in product(allColumnValues1, allColumnValues2)}
    currentCombs = {tuple(r) for r in filteredData[[column1, column2]].to_numpy()}
    diffCombs = {(i, j, None) for i, j in allCombs.symmetric_difference(currentCombs)}
    diffCombsDf = pd.DataFrame(diffCombs, columns=filteredData.columns)
    filteredData = filteredData.append(diffCombsDf, ignore_index=True)

  # Construct holoview table object and sort by column2
  table = hv.Table(filteredData, label=f"Frequency of {column1} by {column2}").sort([column2, column1])
  
  # Use default heatmap options if no option is specified
  heatmapOptions = kwargs
  if (len(kwargs) == 0):
    heatmapOptions = {"cmap": 'viridis', "show_title": True, "tools": ['hover'], 
                      "colorbar": True, "toolbar": 'below', "logx": True, 
                      "width": 690, "height": 550}

  # Create heat map object
  heatmap = table.to.heatmap(kdims=[column2, column1], vdims='Frequency')
  heatmap.opts(**heatmapOptions)
      
  return heatmap


