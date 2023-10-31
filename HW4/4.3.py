import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CustomNeuralNet(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(CustomNeuralNet, self).__init__()
        self.firstLayer = nn.Linear(inputDim, hiddenDim)
        self.secondLayer = nn.Linear(hiddenDim, outputDim)
        self.sigmoidActivation = nn.Sigmoid()
        self.softmaxActivation = nn.Softmax(dim=1)

    def forward(self, tensorInput):
        tensorInput = self.firstLayer(tensorInput)
        tensorInput = self.sigmoidActivation(tensorInput)
        tensorInput = self.secondLayer(tensorInput)
        tensorInput = self.softmaxActivation(tensorInput)
        return tensorInput

dataTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnistTrainDataset = torchvision.datasets.MNIST(root='./mnistData', train=True, download=True, transform=dataTransform)
trainDataLoader = torch.utils.data.DataLoader(mnistTrainDataset, batch_size=64, shuffle=True)

mnistTestDataset = torchvision.datasets.MNIST(root='./mnistData', train=False, download=True, transform=dataTransform)
testDataLoader = torch.utils.data.DataLoader(mnistTestDataset, batch_size=64, shuffle=False)

inputDimension = 28*28
hiddenDimension = 300
outputDimension = 10
learningRate = 0.001
trainingEpochs = 5

neuralModel = CustomNeuralNet(inputDimension, hiddenDimension, outputDimension)
lossFunction = nn.CrossEntropyLoss()
optimizerFunction = optim.Adam(neuralModel.parameters(), lr=learningRate)

epochLosses = []
for epoch in range(trainingEpochs):
    for dataBatchIndex, (imageBatch, labelBatch) in enumerate(trainDataLoader):
        imageBatch = imageBatch.reshape(-1, inputDimension)
        
        predictedOutputs = neuralModel(imageBatch)
        batchLoss = lossFunction(predictedOutputs, labelBatch)
        
        optimizerFunction.zero_grad()
        batchLoss.backward()
        optimizerFunction.step()
    
    epochLosses.append(batchLoss.item())

neuralModel.eval()
with torch.no_grad():
    correctlyPredicted = 0
    totalSamples = 0
    for imageBatch, labelBatch in testDataLoader:
        imageBatch = imageBatch.reshape(-1, inputDimension)
        predictedOutputs = neuralModel(imageBatch)
        _, predictedLabels = torch.max(predictedOutputs.data, 1)
        totalSamples += labelBatch.size(0)
        correctlyPredicted += (predictedLabels == labelBatch).sum().item()

calcTestError = (totalSamples - correctlyPredicted) / totalSamples
print('Test error of the network on the 10000 test images: {:.2f}'.format(calcTestError))

plt.plot(epochLosses)
plt.xlabel('Training Epochs')
plt.ylabel('Epoch Loss')
plt.title('Neural Network Learning Curve')
plt.savefig("4.3.png")
plt.show()
