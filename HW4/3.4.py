import os

folder_path = 'languageID'

# Initialize the bag-of-words count vector
char_counts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

# Count characters in the test document
test_file_path = os.path.join(folder_path, 'e10.txt')
with open(test_file_path, 'r', encoding="utf-8") as f:
    content = f.read()
    for char in content:
        if char in char_counts:
            char_counts[char] += 1

# Convert dictionary to a list to represent the vector
bow_vector = list(char_counts.values())

print("Bag-of-words vector for e10.txt:", bow_vector)
# for char in bow_vector.keys():
#     print("$$", char,":", bow_vector[char], "$$")