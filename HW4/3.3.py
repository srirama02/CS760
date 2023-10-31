import os

langID_directory = 'languageID'

englishCharacterCounts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
japaneseCharacterCounts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
spanishCharacterCounts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

def tallyLanguageCharacters(filepath, countsDictionary):
    with open(filepath, 'r', encoding="utf-8") as f:
        contentData = f.read()
        for character in contentData:
            if character in countsDictionary:
                countsDictionary[character] += 1

for file in os.listdir(langID_directory):
    if file.endswith(".txt") and int(file[1:-4]) < 10:
        fullPath = os.path.join(langID_directory, file)
        if file.startswith('e'):
            tallyLanguageCharacters(fullPath, englishCharacterCounts)
        elif file.startswith('j'):
            tallyLanguageCharacters(fullPath, japaneseCharacterCounts)
        elif file.startswith('s'):
            tallyLanguageCharacters(fullPath, spanishCharacterCounts)

smoothingAlpha = 0.5
possibleChars = 27

englishTheta = {char: (count + smoothingAlpha) / (sum(englishCharacterCounts.values()) + smoothingAlpha * possibleChars) for char, count in englishCharacterCounts.items()}
japaneseTheta = {char: round((count + smoothingAlpha) / (sum(japaneseCharacterCounts.values()) + smoothingAlpha * possibleChars), 4) for char, count in japaneseCharacterCounts.items()}
spanishTheta = {char: round((count + smoothingAlpha) / (sum(spanishCharacterCounts.values()) + smoothingAlpha * possibleChars), 4) for char, count in spanishCharacterCounts.items()}
