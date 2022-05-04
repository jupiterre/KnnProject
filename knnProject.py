

import numpy as np
import pandas as pd
from openpyxl import load_workbook
import csv
from collections import OrderedDict
import pprint
import math

def distance(x, y):
  sumSquaredDiff = 0
  for i in range(0, len(x) - 1):
    squaredDiff= (x[i] - y[i])**2
    sumSquaredDiff = sumSquaredDiff + squaredDiff
  sqrtSumSquaredDiff = sumSquaredDiff**0.5
  return sqrtSumSquaredDiff
  

#y is training data, x is what we're trying to find distance from
def kSmallestDistances(x, y, k):
  allDistances = OrderedDict()
  for n in y:
    dist = distance(x, n)
    allDistances[dist] = n[len(n) - 1]
  allDistances = OrderedDict(sorted(allDistances.items(), key=lambda item: item[0]))
  dict_items = allDistances.items()

  return list(dict_items)[:k]

def findMajorityClass(z):
  classInstances = OrderedDict()
  for i in z:
    if (i[1] in classInstances):
      classInstances[i[1]] = classInstances[i[1]] + 1
    else:
      classInstances[i[1]] = 1
  classInstances = OrderedDict(sorted(classInstances.items(), key=lambda item: item[1]))
  dict_items = classInstances.items()
  majClass = list(dict_items)[-1]

  return majClass[0]

def findMajorityClasses(training, testing, k):
  truepositive = 0
  truenegative = 0
  falsepositive = 0
  falsenegative = 0
  for test in testing:
    predictedClass = findMajorityClass(kSmallestDistances(test, training, k))
    actualClass = test[len(test) - 1]
    if (predictedClass == 1  and actualClass == 1):
      truepositive = truepositive + 1
    elif (predictedClass == 1 and actualClass == -1):
      falsepositive = falsepositive + 1
    elif (predictedClass == -1 and actualClass == -1):
      truenegative = truenegative + 1
    elif (predictedClass == -1 and actualClass == 1):
      falsenegative = falsenegative + 1
    else:
      print("ERROR: predicted class is " + str(predictedClass) + ", actual class is " + str(actualClass))
  return findAccSensSpecPrec(truepositive, truenegative, falsepositive, falsenegative)

def findAccSensSpecPrec(truepos, trueneg, falsepos, falseneg):
  accuracy = float(truepos + trueneg) / (truepos + trueneg + falsepos + falseneg)
  sensitivity = float(truepos) / (truepos + falseneg)
  specificity = float(trueneg) / (falsepos + trueneg)
  if (truepos + falsepos != 0):
    precision = float(truepos) / (truepos + falsepos)
    precision = "{:.5f}".format(precision)
  else:
    precision = "N/A"

  #print("actual vals: " + str(accuracy) + ", " + str(sensitivity) + ", " + str(specificity) + ", " + str(precision))

  #print(float(truepos + trueneg) / (truepos + trueneg + falsepos + falseneg))
  acc = "{:.5f}".format(accuracy)
  sens = "{:.5f}".format(sensitivity)
  spec = "{:.5f}".format(specificity)

  #performance = [accuracy, sensitivity, specificity, precision]
  performance = [acc, sens, spec, precision]
  #print("performance[0]" + str(performance[0]))
  return performance

#data from https://www.kaggle.com/datasets/teertha/personal-loan-modeling
trainingFile = "loanTraining.csv"
testingFile = "loanTesting.csv"

trainingData = np.genfromtxt(trainingFile, delimiter=',')
testingData = np.genfromtxt(testingFile, delimiter=',')

rows = len(trainingData) + len(testingData)
performanceK01 = findMajorityClasses(trainingData, testingData, math.ceil(rows*.01))
performanceK025 = findMajorityClasses(trainingData, testingData, math.ceil(rows*.025))
performanceK05 = findMajorityClasses(trainingData, testingData, math.ceil(rows*.05))

performanceK5 = findMajorityClasses(trainingData, testingData, 5)
performanceK10 = findMajorityClasses(trainingData, testingData, 10)
performanceK20 = findMajorityClasses(trainingData, testingData, 20)


print("Experiment\t\tAccuracy\tSensitivity\tSpecificity\tPrecision")
print("--------------------------------------------------------------------------------")
print("dataset testing (k=5) " + "\t" + str(performanceK5[0]) + "\t\t" + str(performanceK5[1]) + "\t\t" + str(performanceK5[2]) + "\t\t" + str(performanceK5[3]))
print("dataset testing (k=10) " + "\t" + str(performanceK10[0]) + "\t\t" + str(performanceK10[1]) + "\t\t" + str(performanceK10[2]) + "\t\t" + str(performanceK10[3]))
print("dataset testing (k=20) " + "\t" + str(performanceK20[0]) + "\t\t" + str(performanceK20[1]) + "\t\t" + str(performanceK20[2]) + "\t\t" + str(performanceK20[3]))

print("dataset testing (k=" + str(math.ceil(rows*.01)) + ") " + "\t" + str(performanceK01[0]) + "\t\t" + str(performanceK01[1]) + "\t\t" + str(performanceK01[2]) + "\t\t" + str(performanceK01[3]))
print("dataset testing (k=" + str(math.ceil(rows*.025)) + ") " + "\t" + str(performanceK025[0]) + "\t\t" + str(performanceK025[1]) + "\t\t" + str(performanceK025[2]) + "\t\t" + str(performanceK025[3]))
print("dataset testing (k=" + str(math.ceil(rows*.05)) + ") " + "\t" + str(performanceK05[0]) + "\t\t" + str(performanceK05[1]) + "\t\t" + str(performanceK05[2]) + "\t\t" + str(performanceK05[3]))
