import os

# Define your vocabulary
vocabulary = 'abcdefghijklmnopqrstuvwxyz '

# Initialize character counts for each language
char_count = {
    'e': {},
    'j': {},
    's': {},
}

# Read and process the training data
for language in ['e', 'j', 's']:
    for i in range(10):
        with open(os.path.join('languageID', f'{language}{i}.txt'), 'r', encoding='utf-8') as file:
            text = file.read().lower()  # Read and convert to lowercase
            for char in text:
                if char in vocabulary:
                    if char in char_count[language]:
                        char_count[language][char] += 1
                    else:
                        char_count[language][char] = 1

print(char_count['e'])

# Number of documents in each class
num_docs_in_class = {language: 10 for language in ['e', 'j', 's']}
print(num_docs_in_class)

# Total number of documents
total_num_docs = sum(num_docs_in_class.values())

# Size of the vocabulary
vocabulary_size = len(vocabulary)

# Initialize dictionary to store prior probabilities
prior_probabilities = {}

# Calculate and store the prior probabilities
for language in ['e', 'j', 's']:
    prior_probabilities[language] = (num_docs_in_class[language] + 0.5) / (total_num_docs + 0.5 * vocabulary_size)

# Print and include the prior probabilities in your final report
for language, probability in prior_probabilities.items():
    print(f'P(Y={language}) = {probability}')
