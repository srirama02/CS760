import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class UniqueNeuralNetwork(nn.Module):
    def __init__(self, uniqueInputSize, uniqueHiddenSize, uniqueOutputSize):
        super(UniqueNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(uniqueInputSize, uniqueHiddenSize)
        self.layer2 = nn.Linear(uniqueHiddenSize, uniqueOutputSize)
        self.activationSigmoid = nn.Sigmoid()
        self.activationSoftmax = nn.Softmax(dim=1)

    def forward(self, tensorX):
        tensorX = self.layer1(tensorX)
        tensorX = self.activationSigmoid(tensorX)
        tensorX = self.layer2(tensorX)
        tensorX = self.activationSoftmax(tensorX)
        return tensorX

dataTransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainingData = torchvision.datasets.MNIST(root='./uniqueData', train=True, download=True, transform=dataTransforms)
trainingLoader = torch.utils.data.DataLoader(trainingData, batch_size=64, shuffle=True)

testingData = torchvision.datasets.MNIST(root='./uniqueData', train=False, download=True, transform=dataTransforms)
testingLoader = torch.utils.data.DataLoader(testingData, batch_size=64, shuffle=False)

uniqueInputSize = 28*28
uniqueHiddenSize = 300
uniqueOutputSize = 10
uniqueLearningRate = 0.001
uniqueNumEpochs = 5

uniqueModel = UniqueNeuralNetwork(uniqueInputSize, uniqueHiddenSize, uniqueOutputSize)
uniqueCriterion = nn.CrossEntropyLoss()
uniqueOptimizer = optim.Adam(uniqueModel.parameters(), lr=uniqueLearningRate)

def initializeZeroWeights(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.fill_(0)
        layer.bias.data.fill_(0)

def initializeRandomWeights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight, -1, 1)
        layer.bias.data.fill_(0)

def executeTraining(modelInstance, filenameSave):
    listOfLosses = []
    for uniqueEpoch in range(uniqueNumEpochs):
        for index, (imageBatch, labelBatch) in enumerate(trainingLoader):
            imageBatch = imageBatch.reshape(-1, uniqueInputSize)
            outputPredicted = modelInstance(imageBatch)
            lossCurrent = uniqueCriterion(outputPredicted, labelBatch)
            
            uniqueOptimizer.zero_grad()
            lossCurrent.backward()
            uniqueOptimizer.step()
        
        listOfLosses.append(lossCurrent.item())

    modelInstance.eval()
    with torch.no_grad():
        countCorrect = 0
        countTotal = 0
        for imageBatch, labelBatch in testingLoader:
            imageBatch = imageBatch.reshape(-1, uniqueInputSize)
            outputPredicted = modelInstance(imageBatch)
            _, labelPredicted = torch.max(outputPredicted.data, 1)
            countTotal += labelBatch.size(0)
            countCorrect += (labelPredicted == labelBatch).sum().item()

    errorTest = (countTotal - countCorrect) / countTotal
    print(f'Test error using {filenameSave}: {errorTest:.2f}')

    plt.plot(listOfLosses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve ({filenameSave})')
    plt.savefig(filenameSave)
    plt.show()

uniqueModel.apply(initializeZeroWeights)
executeTraining(uniqueModel, "4.3.a.png")

uniqueModel.apply(initializeRandomWeights)
executeTraining(uniqueModel, "4.3.b.png")
