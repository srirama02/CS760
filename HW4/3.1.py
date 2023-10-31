import os

folder_path = 'languageID'

char_count_e = 0
char_count_j = 0
char_count_s = 0

def count_chars(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read()
        return sum(1 for char in content if char.islower() or char == ' ')

for filename in os.listdir(folder_path):
    if filename.endswith(".txt") and int(filename[1:-4]) < 10:
        filepath = os.path.join(folder_path, filename)
        if filename.startswith('e'):
            char_count_e += count_chars(filepath)
        elif filename.startswith('j'):
            char_count_j += count_chars(filepath)
        elif filename.startswith('s'):
            char_count_s += count_chars(filepath)

total_char_count = char_count_e + char_count_j + char_count_s
alpha = 0.5
K = 3

p_e = (char_count_e + alpha) / (total_char_count + alpha * K)
p_j = (char_count_j + alpha) / (total_char_count + alpha * K)
p_s = (char_count_s + alpha) / (total_char_count + alpha * K)

print(f"P(y=e): {p_e}")
print(f"P(y=j): {p_j}")
print(f"P(y=s): {p_s}")