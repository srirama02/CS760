import os

folder_path = 'languageID'

# Dictionaries to hold character counts for English, Japanese, and Spanish
char_counts_e = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
char_counts_j = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
char_counts_s = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

# Function to count valid characters in a file for a given language
def count_chars_for_language(filepath, char_counts):
    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read()
        for char in content:
            if char in char_counts:
                char_counts[char] += 1

# Count characters for each language in the training data (0.txt to 9.txt)
for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and int(filename[1:-4]) < 10:
        filepath = os.path.join(folder_path, filename)
        if filename.startswith('e'):
            count_chars_for_language(filepath, char_counts_e)
        elif filename.startswith('j'):
            count_chars_for_language(filepath, char_counts_j)
        elif filename.startswith('s'):
            count_chars_for_language(filepath, char_counts_s)

alpha = 0.5
K = 27  # Number of valid characters (a-z and space)

# Compute class conditional probabilities for each character in each language
theta_e = {char: (count + alpha) / (sum(char_counts_e.values()) + alpha * K) for char, count in char_counts_e.items()}
theta_j = {char: round((count + alpha) / (sum(char_counts_j.values()) + alpha * K), 4) for char, count in char_counts_j.items()}
theta_s = {char: round((count + alpha) / (sum(char_counts_s.values()) + alpha * K), 4) for char, count in char_counts_s.items()}

# print("Theta for English:", theta_e)
# print("Theta for Japanese:", theta_j)
# print("Theta for Spanish:", theta_s)

print(list(theta_j.values()))
print()
print(list(theta_s.values()))


# for char in theta_j.keys():
#     print("$$", char,":", theta_j[char], "$$")

# for char in theta_s.keys():
#     print("$$", char,":", theta_s[char], "$$")
