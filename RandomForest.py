import pandas as pd
import numpy as np
import math
from random import randrange


numberStumps =30
outputClassCount = 2
trainingInput = None
trainingOutput = None

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.dimensionSplit = None
        self.valSplit = None
        self.infoGain = None

# Reading csv files
def readDataFromCsv(inp,out):
  xInput = pd.read_csv(inp,header=None).as_matrix()
  yOutput = pd.read_csv(out,header=None).as_matrix()
  return (xInput,yOutput )

# Creating bootstrapped dataset
def bootstrappedTrainingSet(dataPointCount, appendTraining):
    i = 0
    bootstrappedTraining = None
    while i<dataPointCount:
        rowIndex = randrange(0,dataPointCount-1)
        if i == 0:
            bootstrappedTraining = (appendTraining[rowIndex]).reshape(1,appendTraining.shape[1])
        else:
            bootstrappedTraining = np.append(bootstrappedTraining,(appendTraining[rowIndex]).reshape(1,appendTraining.shape[1]), axis = 0)
        i+=1
    return bootstrappedTraining

# list of features based on which we want to split    
def getRandomFeatureList(numberFeatureEvalPerNode, allDimIndex):
    k = 0
    randomFeatureList = list()
    while k<numberFeatureEvalPerNode:
        index = randrange(0,len(allDimIndex)-1)
        if index not in randomFeatureList:
            randomFeatureList.append(index)
        k+=1
    return randomFeatureList


def calculateProbArrayForData(inputData, outputClassCount):
    probArray = np.zeros(outputClassCount)
    yIndex = inputData.shape[1]-1
    yOutputColumn = inputData[:,yIndex]
    dataPointCount = inputData.shape[0]
    for classC in range(outputClassCount):
        probArray[classC] = (1.0*(yOutputColumn == classC).sum())/(dataPointCount)
    return probArray

#Splitting the data based in dimension value
def splitDataBasedVal(data, dimIndex, corresVal):
    left = None
    right = None
    for index in range(data.shape[0]):
        rowToAppend = data[index].reshape(1,data[index].shape[0]) 
        if data[index][dimIndex]< corresVal:
            if left is None:
                left = rowToAppend
            else:
                left = np.append(left,rowToAppend, axis = 0)
        else:
            if right is None:
                right = rowToAppend
            else:
                right = np.append(right,rowToAppend, axis = 0)
    return (left, right)

# Method to calculate entropy
def calculateEntropy(nodeProbArray, outputClassCount):
    entropy = 0.0
    for c in range(outputClassCount):
        if nodeProbArray[c]>0:
            entropy-=nodeProbArray[c]*math.log2(nodeProbArray[c])
    return entropy

# Calculate information gain upon splitting         
def calculateInformationGainAndTree(bootstrappedData, dimIndex, corresValToSplit, outputClassParentProbArray, outputClassCount):
    left, right = splitDataBasedVal(bootstrappedData,dimIndex,corresValToSplit)
    if left is not None:
        probArrayLeft = calculateProbArrayForData(left, outputClassCount)
        probGoLeftChild = 1.0*(left.shape[0])/(bootstrappedData.shape[0])
    else:
        probArrayLeft = np.zeros(outputClassCount) 
        probGoLeftChild = 0.0
        
    if right is not None:
        probArrayRight = calculateProbArrayForData(right, outputClassCount)
        probGoRightChild = 1.0*(right.shape[0])/(bootstrappedData.shape[0])
    else:
        probArrayRight= np.zeros(outputClassCount)
        probGoRightChild = 0.0
    
    parentEntropy = calculateEntropy(outputClassParentProbArray, outputClassCount)
    leftChildEntropy = calculateEntropy(probArrayLeft,outputClassCount)
    rightChildEntropy = calculateEntropy(probArrayRight,outputClassCount)
    
    totalChildEntropy = probGoLeftChild*leftChildEntropy + probGoRightChild*rightChildEntropy
    infoGainSplit = parentEntropy-totalChildEntropy
    
    root = Tree()
    root.left = probArrayLeft
    root.right = probArrayRight
    root.data = outputClassParentProbArray
    root.dimensionSplit = dimIndex
    root.valSplit = corresValToSplit
    root.infoGain = infoGainSplit
    
    return root
    
    
    
def getMaximumGainTree(bootstrappedData, featureIndexperNode, outputClassParentProbArray, outputClassCount):
    dimToEvaluate = len(featureIndexperNode)
    valuesPerDimension = 2 # HARD-CODED
    k = 0
    maxInfoGain = 0.0
    finalPredictorTree = None
    print("dimToEvaluate, ", dimToEvaluate)
    while k<dimToEvaluate:
        dimIndex = featureIndexperNode[k]
        z = 0
        while z<valuesPerDimension:
            randomRowToSplit = bootstrappedData[randrange(0,bootstrappedData.shape[0])]
            corresValToSplit = randomRowToSplit[dimIndex]
            treeRoot = calculateInformationGainAndTree(bootstrappedData, dimIndex, corresValToSplit, outputClassParentProbArray, outputClassCount)
            if k==0 and z==0:
                finalPredictorTree = treeRoot
                maxInfoGain = treeRoot.infoGain
            else:
                if maxInfoGain<treeRoot.infoGain:
                    finalPredictorTree = treeRoot
                    maxInfoGain = treeRoot.infoGain
            z=+1
        k+=1   
        
    return finalPredictorTree

def predictClass(tree, dataRow):
    dimToUse = tree.dimensionSplit
    threshValue = tree.valSplit
    if dataRow[dimToUse]< threshValue:
        return tree.left
    else:
        return tree.right

def predictForRow(baggedTrees, dataRow):
    probMatrix = None
    for treeN in range(len(baggedTrees)):
        tree = baggedTrees[treeN]
        probArray = (predictClass(tree, dataRow)).reshape(1,outputClassCount)
        if probMatrix is None:
            probMatrix = probArray
        else:
            probMatrix = np.append(probMatrix,probArray, axis = 0)
    prob = 1.0*np.sum(probMatrix, axis = 0)/len(baggedTrees)
    return np.argmax(prob)

def randomForestPrediction(data, baggedTrees):
    predictions = [predictForRow(baggedTrees, row) for row in data]
    return predictions


        
trainingInput, trainingOutput = readDataFromCsv('train_X_dog_cat.csv','train_y_dog_cat.csv')
trainingOutput[trainingOutput == -1] = 0 # Setting output of -1 to 0
dataPointCount = trainingInput.shape[0]
dimensionCount = trainingInput.shape[1]

#numberFeatureEvalPerNode = int(math.sqrt(dimensionCount))
numberFeatureEvalPerNode = 1
allDimIndex = list(range(dimensionCount))

appendTraining = np.append(trainingInput, trainingOutput, axis = 1)


## Training numberStumps decisionStumps
baggedTrees = list()
for k in range(numberStumps):
    bootstrapSet = bootstrappedTrainingSet(dataPointCount,appendTraining)
    featureIndexPerNode = getRandomFeatureList(numberFeatureEvalPerNode,allDimIndex)
    outputClassParentProbArray = calculateProbArrayForData(bootstrapSet, outputClassCount)
    root = getMaximumGainTree(bootstrapSet, featureIndexPerNode, outputClassParentProbArray, outputClassCount)
    baggedTrees.append(root)
    
for h in range(len(baggedTrees)):
    print(baggedTrees[h].infoGain)


## Training Error
predictionForData = randomForestPrediction(appendTraining, baggedTrees)
predictionTrainingArray = np.asarray(predictionForData)
predictionTrainingArray = predictionTrainingArray.reshape(predictionTrainingArray.shape[0],1)
diffArrayTrain = predictionTrainingArray-trainingOutput
misClassificationsTrain =  (diffArrayTrain!=0).sum()
print("Training error ",1.0*(misClassificationsTrain/dataPointCount))


##Testing Error
testInput, testOutput = readDataFromCsv('test_X_dog_cat.csv', 'test_y_dog_cat.csv')
testOutput[testOutput==-1] = 0
appendedTesting = np.append(testInput, testOutput, axis = 1)

predictionForData = randomForestPrediction(appendedTesting, baggedTrees)
predictionTestingArray = np.asarray(predictionForData)
predictionTestingArray = predictionTestingArray.reshape(predictionTestingArray.shape[0],1)
diffArrayTest = predictionTestingArray-testOutput
misClassificationsTest =  (diffArrayTest!=0).sum()
print("Testing  error ",1.0*(misClassificationsTest/testOutput.shape[0]))


   
    
    


