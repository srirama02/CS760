import os

folder_path_langID = 'languageID'

char_count_english = 0
char_count_japanese = 0
char_count_spanish = 0

def count_characters_in_file(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read()
        return sum(1 for char in content if char.islower() or char == ' ')

for filename in os.listdir(folder_path_langID):
    if filename.endswith(".txt") and int(filename[1:-4]) < 10:
        filepath = os.path.join(folder_path_langID, filename)
        if filename.startswith('e'):
            char_count_english += count_characters_in_file(filepath)
        elif filename.startswith('j'):
            char_count_japanese += count_characters_in_file(filepath)
        elif filename.startswith('s'):
            char_count_spanish += count_characters_in_file(filepath)

total_char_count_all = char_count_english + char_count_japanese + char_count_spanish

alpha_smoothing = 0.5
num_languages = 3

p_english = (char_count_english + alpha_smoothing) / (total_char_count_all + alpha_smoothing * num_languages)
p_japanese = (char_count_japanese + alpha_smoothing) / (total_char_count_all + alpha_smoothing * num_languages)
p_spanish = (char_count_spanish + alpha_smoothing) / (total_char_count_all + alpha_smoothing * num_languages)
