import os
import math

langIDDirectory = 'languageID'

englishCharCounts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
japaneseCharCounts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
spanishCharCounts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
testCharCounts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

def tallyLanguageChars(filePath, countsDict):
    with open(filePath, 'r', encoding="utf-8") as f:
        fileContent = f.read()
        for character in fileContent:
            if character in countsDict:
                countsDict[character] += 1

for fileName in os.listdir(langIDDirectory):
    if fileName.endswith(".txt") and int(fileName[1:-4]) < 10:
        fullPath = os.path.join(langIDDirectory, fileName)
        if fileName.startswith('e'):
            tallyLanguageChars(fullPath, englishCharCounts)
        elif fileName.startswith('j'):
            tallyLanguageChars(fullPath, japaneseCharCounts)
        elif fileName.startswith('s'):
            tallyLanguageChars(fullPath, spanishCharCounts)

smoothingAlpha = 0.5
possibleCharCount = 27

englishTheta = {char: (count + smoothingAlpha) / (sum(englishCharCounts.values()) + smoothingAlpha * possibleCharCount) for char, count in englishCharCounts.items()}
japaneseTheta = {char: (count + smoothingAlpha) / (sum(japaneseCharCounts.values()) + smoothingAlpha * possibleCharCount) for char, count in japaneseCharCounts.items()}
spanishTheta = {char: (count + smoothingAlpha) / (sum(spanishCharCounts.values()) + smoothingAlpha * possibleCharCount) for char, count in spanishCharCounts.items()}

testFilePath = os.path.join(langIDDirectory, 'e10.txt')
with open(testFilePath, 'r', encoding="utf-8") as f:
    fileContent = f.read()
    for character in fileContent:
        if character in testCharCounts:
            testCharCounts[character] += 1

testCharVector = list(testCharCounts.values())

def computeLogProbability(x, languageTheta):
    logProb = 0
    for i, xi in enumerate(x):
        charKey = list(testCharCounts.keys())[i]
        logProb += xi * math.log(languageTheta[charKey])
    return logProb

probEnglish = computeLogProbability(testCharVector, englishTheta)
probJapanese = computeLogProbability(testCharVector, japaneseTheta)
probSpanish = computeLogProbability(testCharVector, spanishTheta)
