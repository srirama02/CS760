import os
import math

folder_path = 'languageID'

# Initialize the bag-of-words count vector for each language and the test document
char_counts_e = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
char_counts_j = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
char_counts_s = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
char_counts_test = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

def count_chars_for_language(filepath, char_counts):
    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read()
        for char in content:
            if char in char_counts:
                char_counts[char] += 1

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

theta_e = {char: (count + alpha) / (sum(char_counts_e.values()) + alpha * K) for char, count in char_counts_e.items()}
theta_j = {char: (count + alpha) / (sum(char_counts_j.values()) + alpha * K) for char, count in char_counts_j.items()}
theta_s = {char: (count + alpha) / (sum(char_counts_s.values()) + alpha * K) for char, count in char_counts_s.items()}

# Count characters in the test document
test_file_path = os.path.join(folder_path, 'e10.txt')
with open(test_file_path, 'r', encoding="utf-8") as f:
    content = f.read()
    for char in content:
        if char in char_counts_test:
            char_counts_test[char] += 1

# Convert dictionary to a list to represent the vector
x_vector = list(char_counts_test.values())

# Compute log(p(x | y)) for each language
def compute_log_probability_given_language(x, theta):
    log_probability = 0
    for i, xi in enumerate(x):
        char = list(char_counts_test.keys())[i]
        if theta[char] > 0:  # Ensure the character probability is not zero before taking its logarithm
            log_probability += xi * math.log(theta[char])
    return log_probability  # Keep the probability in log space

log_p_x_given_e = compute_log_probability_given_language(x_vector, theta_e)
log_p_x_given_j = compute_log_probability_given_language(x_vector, theta_j)
log_p_x_given_s = compute_log_probability_given_language(x_vector, theta_s)

print("Log-probability of x given English:", log_p_x_given_e)
print("Log-probability of x given Japanese:", log_p_x_given_j)
print("Log-probability of x given Spanish:", log_p_x_given_s)
