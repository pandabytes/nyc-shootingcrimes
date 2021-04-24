import json
import plotly
import time

import pandas as pd
import ipyleaflet as lf
import ipywidgets as widgets
import plotly.express as px

from branca.colormap import linear
from geopy.geocoders import Nominatim

def clearLayers(worldMap: lf.Map, layerType: type, dispose: bool = True) -> int:
  ''' Remove all layers based on the given layerType 
      @worldMap: Map object
      @layerType: type of layer to be removed
      @dispose: if True, delete the removed layer
      Returns: number of removed layers
  '''
  layers = list(filter(lambda layer: isinstance(layer, layerType), worldMap.layers))
  map(lambda layer: worldMap.remove_layer(layer), layers)
  
  if (dispose):
    for layer in layers:
      del layer

  return len(list(layers))

def removeLayerByType(worldMap: lf.Map, layerName: str, layerType: type, dispose: bool = True) -> bool:
  ''' Remove a heatmap layer by its name 
      @worldMap: Map object
      @layerName: name of the layer
      @layerType: type of the layer
      @dispose: if True, delete the removed layer
      Returns: True if removed, False if not
  '''
  layers = list(filter(lambda layer: isinstance(layer, layerType) and layer.name == layerName, worldMap.layers))
  map(lambda layer: worldMap.remove_layer(layer), layers)

  if (dispose):
    for layer in layers:
      del layer

  return len(layers) > 0

def plotHeatmap(locations: [tuple], heatmapName: str, 
                worldMap: lf.Map = None, center: tuple = (0, 0), **kwargs) -> lf.Map:
  ''' Plot the given lat lon locations as a heatmap layer on a world map object
      @center: the center coordinate when creating the world map object
      @locations: list of latitude & longitude pair
      @worldMap: ipyleaflet Map object. If this is specified, the new heatmap will be addded
                 to this object. Else a new Map object will be created
      @heatmapName: name of the heatmap layer
      Returns: a newly created Map object or the passed in worldMap object.
  '''
  # Create map if it's not passed in to the function
  if (worldMap is None):
    baseMap = lf.basemaps.CartoDB.DarkMatter # lf.basemaps.Esri.WorldTopoMap
    worldMap = lf.Map(basemap=baseMap, center=center, zoom=10)
    worldMap.add_control(lf.FullScreenControl())
    worldMap.add_control(lf.LayersControl(position='topright'))
      
  # Remove existing heatmap layer that has the same name
  removeLayerByType(worldMap, heatmapName, lf.Heatmap)

  # Add the heatmap layer to the world map object
  heatmap = lf.Heatmap(locations=locations, radius=20, name=heatmapName, **kwargs)
  worldMap.add_layer(heatmap)

  return worldMap

def __syncIdToValueDict(geoJson: dict, idToValue: dict) -> dict:
  ''' Sync the idToValue dict to match with the ids in the geo JSON.
      If the idToValue doesn't contain ids that are in geo JSON, then
      those ids will be assigned with 0.
      @geoJson: 
      @idToValue: a dictionary containing mappings of ids to their associated values
      Returns: an updated idToValue if it contains missing ids or a copy of @idToValue
                if there are no missing ids.
  '''
  # Get all the ids from the geo json data
  geoJsonIdSet = set(feature["id"] for feature in geoJson["features"])

  # Sync the id to match with the id in geo JSON data
  # If there are differences, then update the ids dict with the missing id from
  # geo JSON with value 0
  #
  # Assume that the geo JSON contains ALL ids and the idToValue parameter
  # is a subset of the geo JSON
  updatedIdToValue = dict(idToValue)
  for diff in geoJsonIdSet.difference(set(updatedIdToValue.keys())):
    updatedIdToValue[diff] = 0

  # If there are any ids that exist in updatedIdToValue but DON'T exist
  # in geoJsonIdSet, then remove those ides from updatedIdToValue
  # This forces updatedIdToValue to only contains the ids in geoJsonIdSet
  for diff in set(updatedIdToValue.keys()).difference(geoJsonIdSet):
    del updatedIdToValue[diff]

  return updatedIdToValue

def plotLeaflet(geoJson: dict, idToValue: dict, worldMap: lf.Map = None, 
                includeEmpty: bool = True, labelFormat: str = "Id={0} - Value={1}", 
                mapArgs: dict = {}, heatmapArgs: dict = {}) -> lf.Map:
  ''' Plot the choropleth on the map using the ipyleaflet library
      @idToValue: a dictionary containing mappings of ids to their associated values
      @worldMap: ipyleaflet Map object. If this is specified, the new heatmap will be addded
                 to this object. Else a new Map object will be created
      @includeEmpty: true if we want to display the regions that have value 0. False otherwise
      @labelFormat: the string format of the text that is displayed when the mouse hovers over a polygon.
                    The format must contain exactly 2 placeholders.
      Returns: map object
  '''
  updatedIdToValue = __syncIdToValueDict(geoJson, idToValue)

  # If we don't display empty regions, we need to
  # create a new geo JSON that does NOT contain the 
  # ids that have 0 value
  if (not includeEmpty):
    origGeoJson = geoJson
    geoJson = {"type": "FeatureCollection", "features": []}

    for idValue, value in updatedIdToValue.items():
      if (value > 0):
        feature = getFeatureById(origGeoJson, idValue)
        geoJson["features"].append(feature)

  # Create the world map object if not specified
  if (worldMap is None):
    #basemap=lf.basemaps.CartoDB.DarkMatter, center=center, zoom=10
    worldMap = lf.Map(**mapArgs)
    worldMap.add_control(lf.FullScreenControl())
    worldMap.add_control(lf.LayersControl(position="topright"))

  # Default heatmap arguments
  minVal, maxVal = min(updatedIdToValue.values()), max(updatedIdToValue.values())
  defaultHeatmapArgs = {"key_on": "id", "border_color": "black", "value_min": minVal, "value_max": maxVal, 
                        "style": {'fillOpacity': 0.8, 'dashArray': '5, 5'}, 
                        "hover_style": {'fillColor': 'purple', 'dashArray': '0', 'fillOpacity': 0.5},
                        "name": "Choropleth", "colormap": linear.OrRd_06}

  # Make a copy of the heatmapArgs, because we would add the default arguments
  # to this dict if the default arguments are not specified by the caller. Making
  # a copy prevents modifying the passed in dict object
  heatmapArgs = dict(heatmapArgs)
  for k, v in defaultHeatmapArgs.items():
    if (k not in heatmapArgs):
      heatmapArgs[k] = v

  # Create the choropleth layer
  choroplethLayer = lf.Choropleth(geo_data=geoJson, choro_data=updatedIdToValue, **heatmapArgs)
                        
  # Create a label widget to display the currently hovered polygon id &
  # the value associated with that id
  labelWidget = widgets.Label(value="")
  widgetControl = lf.WidgetControl(widget=labelWidget, position="bottomright")
  worldMap.add_control(widgetControl)

  def mouseout(*args, **kwargs):
    ''' Set the label value to empty string when the mouse exits a region '''
    labelWidget.value = ""
    
  def hover(*args, **kwargs):
    ''' Set the label to the id & its associated value '''
    idValue = kwargs["id"]
    labelWidget.value = labelFormat.format(idValue, updatedIdToValue[idValue])

  # Register callbacks
  choroplethLayer.on_mouseout(mouseout)
  choroplethLayer.on_hover(hover)

  # Remove existing choropleth layer that has the same name
  choroplethName = heatmapArgs["name"] if ("name" in heatmapArgs) else ""
  removeLayerByType(worldMap, choroplethName, lf.Choropleth)

  worldMap.add_layer(choroplethLayer)
  return worldMap

def plotPlotly(geoJson: dict, dataFrame: pd.DataFrame, idColumn: str, valueColumn: str, 
               title: str = "", mapboxStyle: str = "carto-darkmatter", includeEmpty: bool = False,
               colorscale: str = "ylorrd", **kwargs) -> plotly.graph_objs._figure.Figure:
  ''' Plot choropleth using the plotly library '''
  if (includeEmpty):
    featureIds = [feature["id"] for feature in geoJson["features"]]
    missingFeatureIds = list(filter(lambda featureId: featureId not in dataFrame[idColumn].unique(), featureIds))
    
    newRows = [pd.Series([i, 0], index=dataFrame.columns) for i in missingFeatureIds]
    dataFrame = dataFrame.append(newRows, ignore_index=True)

  fig = px.choropleth_mapbox(dataFrame, 
                             geojson=geoJson, 
                             locations=idColumn, 
                             color=valueColumn,
                             color_continuous_scale=colorscale, 
                             mapbox_style=mapboxStyle,
                             **kwargs)

  fig.update_geos(fitbounds="locations", visible=True)
  fig.update_layout(margin={"r":0,"l":0,"b":0}, title_text=title)
  return fig

def getFeatureById(geoJson: dict, idValue: str) -> dict:
  ''' Get a feature from geo JSON data by its id. 
      @geoJson: json data in dict object
      @idValue: the id to find
      Returns: the feature dictionary. None if not found
  '''
  features = list(filter(lambda feature: feature["id"] == idValue, geoJson["features"]))
  return features[0] if (len(features) > 0) else None

def parseGeoJsonFile(geoJsonFilePath: str, pathToFeatureId=["id"]) -> dict:
  ''' Parse the geo json file and return dict object.

      @geoJsonFilePath: path to the geo JSON file
      @pathToFeatureId: list of keys that lead to where the id of a feature is stored
      Returns: dictionary containing the geo JSON
  '''
  if (pathToFeatureId is None or len(pathToFeatureId) == 0):
    raise ValueError("pathToFeatureId must be a list of string with length at least 1.")

  # Read in the geojson file
  geoJson = None
  with open(geoJsonFilePath, "r") as geoFile:
    geoJson = json.loads(geoFile.read())
      
  # Verify that geojson is a FeatureCollection object
  if ("type" not in geoJson or geoJson["type"] != "FeatureCollection"):
    raise ValueError("Geojson is invalid. Geojson needs to have \"type\" set to \"FeatureCollection\"")

  # Add an "id" key so that lf.Choropleth knows
  # how to associate the id to its value
  features = geoJson["features"]
  for feature in features:
    tempObj = feature

    # Now use pathToFeatureId to find where the
    # id is defined in this geojson data. 
    for path in pathToFeatureId:
      tempObj = tempObj[path]

    if (type(tempObj) is not str):
      raise ValueError(f"id value must type string. Got type {type(tempObj)}")

    # Add the "id" key and its value
    feature["id"] = tempObj

  return geoJson

def plotScatterOnMap(dataFrame: pd.DataFrame, latCol: str, lonCol: str, scatterKwargs={}, layoutKwargs={}):
  ''' Plot a scatter plot of points on a map
      @latCol: the column name that contains latitudes
      @lonCol: the column name that contains longitudes
      @scatterKwargs: dict contains additional kw arguments for px.scatter_mapbox
      @scatterKwargs: dict contains additional kw arguments for plotly.graph_objs._figure.Figure.update_layout

      Returns: plotly.graph_objs._figure.Figure object
  '''
  fig = px.scatter_mapbox(dataFrame, lat=latCol, lon=lonCol, **scatterKwargs)
  
  defaultLayoutKwargs = {"margin": {"r": 0, "l": 0, "b" :0}, "mapbox_style": "carto-darkmatter"}
  finalLayoutKwargs = {}

  # Use default arguments first
  # They can be overriden by user specified layout arguments
  for k,v in defaultLayoutKwargs.items():
    finalLayoutKwargs[k] = v

  for k, v in layoutKwargs.items():
    finalLayoutKwargs[k] = v
  
  # Update the layout
  fig.update_layout(**finalLayoutKwargs)
  return fig

def getAddresses(lats: [float], lons: [float], verbose: bool = False) -> [dict]:
  ''' Get a list of addresses given a list of latitudes and longitudes
      Wait for 1 second before processing the next pair of latitude and longitude
      
      @lats: list-like object that contains latitude values   
      @lons: list-like object that contains longitude values
      Returns: list of dictionaries containing parts of an address
               Ex: [{"house_number": 123, "street": "Lincoln Ave", "city": "Irvine", "state": "CA"}]
  '''
  if (len(lats) != len(lons)):
    raise ValueError(f"Number of latitudes must equal to the number of longitudes")

  addresses = []
  geolocator = Nominatim(user_agent="test")

  for i, (lat, lon) in enumerate(zip(lats, lons)):
    location = geolocator.reverse(f"{lat}, {lon}")
    address = location.raw["address"]
    addresses.append(address)

    if (verbose):
      print(f"[{i+1}/{len(lats)}] ({lat}, {lon}) -> {address}")

    time.sleep(1)

  return addresses
