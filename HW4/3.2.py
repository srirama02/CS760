import os

folder_path = 'languageID'

# Dictionary to hold character counts for English
char_counts_e = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

# Function to count valid characters in a file for a given language
def count_chars_for_language(filepath, char_counts):
    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read()
        for char in content:
            if char in char_counts:
                char_counts[char] += 1

# Count characters for English in the training data (0.txt to 9.txt)
for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and int(filename[1:-4]) < 10 and filename.startswith('e'):
        filepath = os.path.join(folder_path, filename)
        count_chars_for_language(filepath, char_counts_e)

# Total character count for English
total_chars_e = sum(char_counts_e.values())
alpha = 0.5
K = 27  # Number of valid characters (a-z and space)

# Compute class conditional probabilities for each character in English
theta_e = {}
for char, count in char_counts_e.items():
    theta_e[char] = round((count + alpha) / (total_chars_e + alpha * K),4)

    # print(char,":", theta_e[char])

print(list(theta_e.values()))
