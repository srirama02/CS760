import os
import math

pathUnique = 'languageID'

charCountsEnglish = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
charCountsJapanese = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
charCountsSpanish = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
charCountsTest = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

def countChars(filepath, charDict):
    with open(filepath, 'r', encoding="utf-8") as fileObj:
        fileContent = fileObj.read()
        for charUnique in fileContent:
            if charUnique in charDict:
                charDict[charUnique] += 1

for file in os.listdir(pathUnique):
    if file.endswith(".txt") and int(file[1:-4]) < 10:
        filePathUnique = os.path.join(pathUnique, file)
        if file.startswith('e'):
            countChars(filePathUnique, charCountsEnglish)
        elif file.startswith('j'):
            countChars(filePathUnique, charCountsJapanese)
        elif file.startswith('s'):
            countChars(filePathUnique, charCountsSpanish)

alphaVal = 0.5
KVal = 27

thetaEnglish = {char: (count + alphaVal) / (sum(charCountsEnglish.values()) + alphaVal * KVal) for char, count in charCountsEnglish.items()}
thetaJapanese = {char: (count + alphaVal) / (sum(charCountsJapanese.values()) + alphaVal * KVal) for char, count in charCountsJapanese.items()}
thetaSpanish = {char: (count + alphaVal) / (sum(charCountsSpanish.values()) + alphaVal * KVal) for char, count in charCountsSpanish.items()}

testFilePath = os.path.join(pathUnique, 'e10.txt')
with open(testFilePath, 'r', encoding="utf-8") as fileObj:
    fileContentTest = fileObj.read()
    for charUnique in fileContentTest:
        if charUnique in charCountsTest:
            charCountsTest[charUnique] += 1

xVec = list(charCountsTest.values())

def computeProb(x, thetaDict):
    logProb = 0
    for idx, xVal in enumerate(x):
        charKey = list(charCountsTest.keys())[idx]
        logProb += xVal * thetaDict[charKey]
    return logProb

probEnglish = computeProb(xVec, thetaEnglish)
probJapanese = computeProb(xVec, thetaJapanese)
probSpanish = computeProb(xVec, thetaSpanish)

priorEnglish = 1/3
priorJapanese = 1/3
priorSpanish = 1/3

postProbEnglish = probEnglish * priorEnglish
postProbJapanese = probJapanese * priorJapanese
postProbSpanish = probSpanish * priorSpanish

totalProbValue = postProbEnglish + postProbJapanese + postProbSpanish
postProbEnglish /= totalProbValue
postProbJapanese /= totalProbValue
postProbSpanish /= totalProbValue

print("Posterior probability of x given English:", postProbEnglish)
print("Posterior probability of x given Japanese:", postProbJapanese)
print("Posterior probability of x given Spanish:", postProbSpanish)

predictionsDict = {'English': postProbEnglish, 'Japanese': postProbJapanese, 'Spanish': postProbSpanish}
predictedLang = max(predictionsDict, key=predictionsDict.get)
