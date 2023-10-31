import os
import math

folder_path = 'languageID'

# Initialize the bag-of-words count vector
char_counts_e = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
char_counts_j = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
char_counts_s = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

def count_chars_for_language(filepath, char_counts):
    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read().lower()
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

theta_e = {char: math.log((count + alpha) / (sum(char_counts_e.values()) + alpha * K)) for char, count in char_counts_e.items()}
theta_j = {char: math.log((count + alpha) / (sum(char_counts_j.values()) + alpha * K)) for char, count in char_counts_j.items()}
theta_s = {char: math.log((count + alpha) / (sum(char_counts_s.values()) + alpha * K)) for char, count in char_counts_s.items()}

# Compute p(x | y) for each language
# def compute_probability_given_language(x, theta):
#     log_probability = 0
#     for i, xi in enumerate(x):
#         char = list(theta.keys())[i]
#         log_probability += xi * theta[char]
#     return math.exp(log_probability)

def compute_probability_given_language(x, theta):
    log_probability = 0
    for i, xi in enumerate(x):
        char = list(theta.keys())[i]
        log_probability += xi * theta[char] 
    return log_probability

def classify_document(file_path):
    char_counts_test = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}
    
    with open(file_path, 'r', encoding="utf-8") as f:
        content = f.read().lower()
        for char in content:
            if char in char_counts_test:
                char_counts_test[char] += 1

    x_vector = list(char_counts_test.values())
    
    p_x_given_e = compute_probability_given_language(x_vector, theta_e)
    p_x_given_j = compute_probability_given_language(x_vector, theta_j)
    p_x_given_s = compute_probability_given_language(x_vector, theta_s)

    posterior_e = p_x_given_e * prior_e
    posterior_j = p_x_given_j * prior_j
    posterior_s = p_x_given_s * prior_s

    predictions = {'English': posterior_e, 'Japanese': posterior_j, 'Spanish': posterior_s}
    return max(predictions, key=predictions.get)

prior_e = 1/3
prior_j = 1/3
prior_s = 1/3

# Initialize the confusion matrix
confusion_matrix = {
    'English': {'English': 0, 'Japanese': 0, 'Spanish': 0},
    'Japanese': {'English': 0, 'Japanese': 0, 'Spanish': 0},
    'Spanish': {'English': 0, 'Japanese': 0, 'Spanish': 0}
}

filename_to_language = {'e': 'English', 'j': 'Japanese', 's': 'Spanish'}


for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and 10 <= int(filename[1:-4]) < 20:
        true_label = filename_to_language[filename[0]]
        predicted_label = classify_document(os.path.join(folder_path, filename))
        confusion_matrix[predicted_label][true_label] += 1

# Print the confusion matrix
print("\nConfusion Matrix:")
for true_label, row in confusion_matrix.items():
    print(f"Predicted: {true_label}")
    for true, count in row.items():
        print(f"\tTrue: {true} - Count: {count}")
