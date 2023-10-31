import os

# Initialize counts
n_e = 0
n_j = 0
n_s = 0
alpha = 0.5
K = 3

# Count documents for each class in the training data (0.txt to 9.txt)
for filename in os.listdir("languageID"):
    if filename.endswith(".txt") and int(filename[1:-4]) < 10:
        if filename.startswith('e'):
            n_e += 1
        elif filename.startswith('j'):
            n_j += 1
        elif filename.startswith('s'):
            n_s += 1

# Total number of documents
N = n_e + n_j + n_s

# Compute prior probabilities with additive smoothing
p_e = (n_e + alpha) / (N + alpha * K)
p_j = (n_j + alpha) / (N + alpha * K)
p_s = (n_s + alpha) / (N + alpha * K)

# Convert to log probabilities (if needed for internal computations)
import math
log_p_e = math.log(p_e)
log_p_j = math.log(p_j)
log_p_s = math.log(p_s)

# Print prior probabilities
print(f"P(y=e): {p_e}")
print(f"P(y=j): {p_j}")
print(f"P(y=s): {p_s}")
