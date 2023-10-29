import numpy as np

def eval_preds(preds, labels):
    # Initialize variables to count true positives, false positives, etc.
    true_pos = np.zeros(np.max(labels) + 1)
    false_pos = np.zeros(np.max(labels) + 1)
    true_neg = np.zeros(np.max(labels) + 1)
    false_neg = np.zeros(np.max(labels) + 1)

    # Count labels and predictions
    for l, p in zip(labels, preds):
        if l == p:
            true_pos[l] += 1
        else:
            false_pos[p] += 1
            false_neg[l] += 1

    # Calculate metrics
    precision = np.sum(true_pos / (true_pos + false_pos + 1e-13)) / len(true_pos)
    recall = np.sum(true_pos / (true_pos + false_neg + 1e-13)) / len(true_pos)
    f1 = 2 * precision * recall / (precision + recall + 1e-13)
    accuracy = np.sum(true_pos) / len(labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }