import os
import collections
import math

# Step 1: Load and Preprocess the Data
def load_data(path):
    data = {}
    for root, _, files in os.walk(path):
        for file in files:
            lang = file[0]  # First character is the class label
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                text = f.read()
                # Tokenize text into characters and filter for a to z and space
                tokens = [c for c in text if c in 'abcdefghijklmnopqrstuvwxyz ']
                data[file] = (lang, tokens)
    return data

# Step 2: Split the Data
def split_data(data, num_training_per_class=10):
    train_data, test_data = [], []
    class_counts = collections.defaultdict(int)
    
    for filename, (lang, tokens) in data.items():
        if class_counts[lang] < num_training_per_class:
            train_data.append((lang, tokens))
            class_counts[lang] += 1
        else:
            test_data.append((lang, tokens))
    
    return train_data, test_data

# Step 3: Estimate Prior Probabilities with Additive Smoothing
def estimate_priors(train_data, alpha=0.5):
    class_counts = collections.defaultdict(int)
    total_count = 0

    for lang, _ in train_data:
        class_counts[lang] += 1
        total_count += 1

    priors = {}
    num_classes = len(class_counts)

    for lang in class_counts:
        priors[lang] = (class_counts[lang] + alpha) / (total_count + alpha * num_classes)

    return priors

# Step 4 and 5: Classify Documents and Make Predictions
def classify_documents(test_data, priors, alpha=0.5):
    predictions = []

    for lang, tokens in test_data:
        class_probs = {}
        for label, prior in priors.items():
            log_prob = math.log(prior)
            for token in tokens:
                class_counts = collections.defaultdict(int)
                for _, train_tokens in train_data:
                    class_counts[label] += train_tokens.count(token)
                token_prob = (class_counts[label] + alpha) / (class_counts[label] + alpha * len(vocabulary))
                log_prob += math.log(token_prob)
            class_probs[label] = log_prob

        predicted_lang = max(class_probs, key=class_probs.get)
        predictions.append((lang, predicted_lang))

    return predictions

# Step 6: Evaluate the Classifier
def evaluate(predictions):
    correct = 0
    total = len(predictions)
    
    for actual, predicted in predictions:
        if actual == predicted:
            correct += 1
    
    accuracy = correct / total
    return accuracy

# Main script
if __name__ == "__main__":
    data = load_data("languageID")
    train_data, test_data = split_data(data, num_training_per_class=10)
    vocabulary = 'abcdefghijklmnopqrstuvwxyz '
    alpha = 0.5
    priors = estimate_priors(train_data, alpha)
    predictions = classify_documents(test_data, priors, alpha)
    accuracy = evaluate(predictions)

    print("Prior Probabilities:")
    for lang, prior in priors.items():
        print(f"P({lang}) = {prior:.4f}")

    print(f"Accuracy: {accuracy:.4f}")
