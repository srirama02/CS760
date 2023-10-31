import os
import math

langDirectory = 'languageID'

englishCharFreqs = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
japaneseCharFreqs = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
spanishCharFreqs = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

def tallyCharacterFrequencies(filePath, charFreqDict):
    with open(filePath, 'r', encoding="utf-8") as f:
        fileContent = f.read().lower()
        for character in fileContent:
            if character in charFreqDict:
                charFreqDict[character] += 1

for file in os.listdir(langDirectory):
    if file.endswith(".txt") and int(file[1:-4]) < 10:
        fullPath = os.path.join(langDirectory, file)
        if file.startswith('e'):
            tallyCharacterFrequencies(fullPath, englishCharFreqs)
        elif file.startswith('j'):
            tallyCharacterFrequencies(fullPath, japaneseCharFreqs)
        elif file.startswith('s'):
            tallyCharacterFrequencies(fullPath, spanishCharFreqs)

smoothingAlpha = 0.5
uniqueCharCount = 27

logThetaEnglish = {char: math.log((count + smoothingAlpha) / (sum(englishCharFreqs.values()) + smoothingAlpha * uniqueCharCount)) for char, count in englishCharFreqs.items()}
logThetaJapanese = {char: math.log((count + smoothingAlpha) / (sum(japaneseCharFreqs.values()) + smoothingAlpha * uniqueCharCount)) for char, count in japaneseCharFreqs.items()}
logThetaSpanish = {char: math.log((count + smoothingAlpha) / (sum(spanishCharFreqs.values()) + smoothingAlpha * uniqueCharCount)) for char, count in spanishCharFreqs.items()}

def calculateLogProbability(x, logTheta):
    totalLogProb = 0
    for i, xi in enumerate(x):
        charKey = list(logTheta.keys())[i]
        totalLogProb += xi * logTheta[charKey] 
    return totalLogProb

def identifyLanguage(filePath):
    testCharFreqs = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
    
    with open(filePath, 'r', encoding="utf-8") as f:
        content = f.read().lower()
        for char in content:
            if char in testCharFreqs:
                testCharFreqs[char] += 1

    charVector = list(testCharFreqs.values())
    
    englishProb = calculateLogProbability(charVector, logThetaEnglish)
    japaneseProb = calculateLogProbability(charVector, logThetaJapanese)
    spanishProb = calculateLogProbability(charVector, logThetaSpanish)

    englishPosterior = englishProb * priorEnglish
    japanesePosterior = japaneseProb * priorJapanese
    spanishPosterior = spanishProb * priorSpanish

    languageProbs = {'English': englishPosterior, 'Japanese': japanesePosterior, 'Spanish': spanishPosterior}
    return max(languageProbs, key=languageProbs.get)

priorEnglish = 1/3
priorJapanese = 1/3
priorSpanish = 1/3

languageMatrix = {
    'English': {'English': 0, 'Japanese': 0, 'Spanish': 0},
    'Japanese': {'English': 0, 'Japanese': 0, 'Spanish': 0},
    'Spanish': {'English': 0, 'Japanese': 0, 'Spanish': 0}
}

fileToLang = {'e': 'English', 'j': 'Japanese', 's': 'Spanish'}

for file in os.listdir(langDirectory):
    if file.endswith(".txt") and 10 <= int(file[1:-4]) < 20:
        actualLang = fileToLang[file[0]]
        predictedLang = identifyLanguage(os.path.join(langDirectory, file))
        languageMatrix[predictedLang][actualLang] += 1
