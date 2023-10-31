import os

folderPathUnique = 'languageID'

alphabetCountUnique = {alphabetItem: 0 for alphabetItem in 'abcdefghijklmnopqrstuvwxyz '}

filePathUnique = os.path.join(folderPathUnique, 'e10.txt')
with open(filePathUnique, 'r', encoding="utf-8") as fileObjUnique:
    fileContentUnique = fileObjUnique.read()
    for charUnique in fileContentUnique:
        if charUnique in alphabetCountUnique:
            alphabetCountUnique[charUnique] += 1

bagOfWordsUnique = list(alphabetCountUnique.values())

print("Bag-of-words vector for e10.txt:", bagOfWordsUnique)
