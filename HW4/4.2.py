import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataMnist = fetch_openml('mnist_784', version=1)
featuresData = dataMnist["data"].values
labelsData = dataMnist["target"].astype(int).values
featuresData = featuresData / 255.0

encoder = OneHotEncoder(sparse=False)
labelsOneHot = encoder.fit_transform(labelsData.reshape(-1, 1))

featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(featuresData, labelsOneHot, test_size=0.2, random_state=42)

def functionSigmoid(inputZ):
    return 1.0 / (1.0 + np.exp(-inputZ))

def functionSigmoidDerivative(inputZ):
    return functionSigmoid(inputZ) * (1.0 - functionSigmoid(inputZ))

def functionSoftmax(inputZ):
    expInputZ = np.exp(inputZ)
    return expInputZ / expInputZ.sum(axis=1, keepdims=True)

def processForwardPropagation(inputFeatures, weights1, weights2):
    zLayer1 = inputFeatures.dot(weights1.T)
    aLayer1 = functionSigmoid(zLayer1)
    zLayer2 = aLayer1.dot(weights2.T)
    aLayer2 = functionSoftmax(zLayer2)
    return zLayer1, aLayer1, zLayer2, aLayer2

def computeFunctionLoss(trueLabels, predictedLabels):
    samplesCount = trueLabels.shape[0]
    lossValue = -np.sum(trueLabels * np.log(predictedLabels)) / samplesCount
    return lossValue

inputNodes = 784
hiddenNodes = 300
outputNodes = 200
rateLearning = 0.1
totalEpochs = 7
sizeBatch = 64

weightsLayer1 = np.random.randn(hiddenNodes, inputNodes) * 0.01
weightsLayer2 = np.random.randn(outputNodes, hiddenNodes) * 0.01

listLosses = []

for epochIndex in range(totalEpochs):
    for batchIndex in range(0, featuresTrain.shape[0], sizeBatch):
        batchFeatures = featuresTrain[batchIndex:batchIndex+sizeBatch]
        batchLabels = labelsTrain[batchIndex:batchIndex+sizeBatch]

        zValue1, aValue1, zValue2, aValue2 = processForwardPropagation(batchFeatures, weightsLayer1, weightsLayer2)

        deltaZ2 = aValue2 - batchLabels
        deltaWeights2 = np.dot(deltaZ2.T, aValue1) / sizeBatch
        deltaALayer1 = np.dot(deltaZ2, weightsLayer2)
        deltaZLayer1 = deltaALayer1 * functionSigmoidDerivative(zValue1)
        deltaWeightsLayer1 = np.dot(deltaZLayer1.T, batchFeatures) / sizeBatch
        
        weightsLayer1 -= rateLearning * deltaWeightsLayer1
        weightsLayer2 -= rateLearning * deltaWeights2

    _, _, _, aValue2Train = processForwardPropagation(featuresTrain, weightsLayer1, weightsLayer2)
    lossTrain = computeFunctionLoss(labelsTrain, aValue2Train)
    listLosses.append(lossTrain)

    _, _, _, aValue2Test = processForwardPropagation(featuresTest, weightsLayer1, weightsLayer2)
    predictedValues = np.argmax(aValue2Test, axis=1)
    trueValues = np.argmax(labelsTest, axis=1)
    accuracyTest = accuracy_score(trueValues, predictedValues)
    errorTest = 1 - accuracyTest
    
    print(f"Epoch {epochIndex + 1}, Test Error: {errorTest:.4f}")

plt.plot(listLosses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.savefig("4.2.png")
plt.show()
