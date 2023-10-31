import os
import math

folder_path = 'languageID'

# Initialize the bag-of-words count vector for the test document
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

# Compute p(x | y) for each language
def compute_probability_given_language(x, theta):
    log_probability = 0
    for i, xi in enumerate(x):
        char = list(char_counts_test.keys())[i]
        log_probability += xi * theta[char] 
    return log_probability

p_x_given_e = compute_probability_given_language(x_vector, theta_e)
p_x_given_j = compute_probability_given_language(x_vector, theta_j)
p_x_given_s = compute_probability_given_language(x_vector, theta_s)


prior_e = 1/3
prior_j = 1/3
prior_s = 1/3

# Compute posterior probabilities
posterior_e = p_x_given_e * prior_e
posterior_j = p_x_given_j * prior_j
posterior_s = p_x_given_s * prior_s

# Normalize the posteriors so they sum to 1
total_prob = posterior_e + posterior_j + posterior_s
posterior_e /= total_prob
posterior_j /= total_prob
posterior_s /= total_prob

print("Posterior probability of x given English:", posterior_e)
print("Posterior probability of x given Japanese:", posterior_j)
print("Posterior probability of x given Spanish:", posterior_s)

# Predict the class label of x
predictions = {'English': posterior_e, 'Japanese': posterior_j, 'Spanish': posterior_s}
predicted_language = max(predictions, key=predictions.get)
print("Predicted class label of x:", predicted_language)
