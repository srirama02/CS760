import matplotlib.pyplot as plt

confidence_scores = [0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1]
true_labels = ['+', '+', '-', '+', '+', '-', '+', '+', '-', '-']

sorted_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)
sorted_scores = [confidence_scores[i] for i in sorted_indices]
sorted_labels = [true_labels[i] for i in sorted_indices]

tpr_list = [0]
fpr_list = [0]
tp = 0
fp = 0
total_positives = true_labels.count('+')
total_negatives = true_labels.count('-')

for i in range(len(sorted_scores)):
    if sorted_labels[i] == '+':
        tp += 1
    else:
        fp += 1
    tpr = tp / total_positives
    fpr = fp / total_negatives
    tpr_list.append(tpr)
    fpr_list.append(fpr)

final_tpr = []
final_fpr = []

for i in [0, 2, 5, 8, 10]:
    final_tpr.append(tpr_list[i])
    final_fpr.append(fpr_list[i])


plt.figure(figsize=(8, 6))
plt.plot(final_fpr, final_tpr, marker='o', linestyle='-', color='b')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid()
plt.savefig('1.5.a.png')
plt.show()