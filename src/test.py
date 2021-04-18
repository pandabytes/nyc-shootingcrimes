import pandas as pd
import matplotlib.pyplot as plt
import models.supervised as sup
import models.utils as ut
from sklearn.model_selection import train_test_split

filePath = r"..\data\NewYorkShootingCrimes.csv"
data = pd.read_csv(filePath, header=0, sep=",")

print(data.head(3))

samples = data[:100]
target = "STATISTICAL_MURDER_FLAG"
xTrain, xTest, yTrain, yTest = train_test_split(samples.drop(target, axis=1), samples[target], test_size=0.3)

def dt():
  dt = sup.DecisionTree(maxDepth=3, numberFeaturesToSplit=5)
  dt.train(xTrain, yTrain)
  preds = dt.classify(xTest)

  classErr = ut.metrics.computeError(preds['Prediction'], yTest) * 100
  print(f"Misclassification error: {classErr:.2f}%")

  # Print feature importance
  print()
  print(dt.featureImportance)
  print()

def rf():
  rf = sup.RandomForest(maxDepth=5, numberDTrees=5, numberFeaturesToSplit=5)
  rf.train(xTrain, yTrain, quiet=True)
  preds = rf.classify(xTest, quiet=True)

  classErr = ut.metrics.computeError(preds['Prediction'], yTest) * 100
  print(f"Misclassification error: {classErr:.2f}%")

  # Print feature importance
  print()
  print(rf.featureImportance)
  print()

# Main
if __name__ == "__main__":
  dt()
  rf()
  
  
