import os

langID_folder_path = 'languageID'

english_char_counts = {char: 0 for char in 'abcdefghijklmnopqrstuvwxyz '}

def tally_characters_in_file(filepath, counts_dictionary):
    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.read()
        for character in content:
            if character in counts_dictionary:
                counts_dictionary[character] += 1

for file_name in os.listdir(langID_folder_path):
    if file_name.endswith(".txt") and int(file_name[1:-4]) < 10 and file_name.startswith('e'):
        full_path = os.path.join(langID_folder_path, file_name)
        tally_characters_in_file(full_path, english_char_counts)

total_english_chars = sum(english_char_counts.values())
smoothing_alpha = 0.5
num_possible_chars = 27

theta_english = {}
for character, count in english_char_counts.items():
    theta_english[character] = round((count + smoothing_alpha) / (total_english_chars + smoothing_alpha * num_possible_chars),4)
