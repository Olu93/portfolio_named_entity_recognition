import numpy as np

def compute_macro_metrics(y_true_list, y_pred_list):
    """
    Macro‑averaged precision, recall, Jaccard, F1 and exact‑match accuracy over documents.
    Returns native Python floats: (precision, recall, jaccard, f1, accuracy)
    """
    precisions, recalls, jaccards, f1s, accuracies = [], [], [], [], []
    for yt, yp in zip(y_true_list, y_pred_list):
        s_true, s_pred = set(yt), set(yp)
        tp = len(s_true & s_pred)

        p = tp / len(s_pred) if s_pred else 0.0
        r = tp / len(s_true) if s_true else 0.0
        j = tp / len(s_true | s_pred) if (s_true | s_pred) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        acc = 1.0 if s_true == s_pred else 0.0

        precisions.append(p)
        recalls.append(r)
        jaccards.append(j)
        f1s.append(f1)
        accuracies.append(acc)

    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(jaccards)),
        float(np.mean(f1s)),
        float(np.mean(accuracies)),
    )

def compute_micro_metrics(y_true_list, y_pred_list):
    """
    Micro‑averaged precision, recall, Jaccard, F1 and exact‑match accuracy across all documents.
    Returns native Python floats: (precision, recall, jaccard, f1, accuracy)
    """
    tp_total = pred_total = true_total = 0
    exact_matches = 0
    n_docs = len(y_true_list)

    for yt, yp in zip(y_true_list, y_pred_list):
        s_true, s_pred = set(yt), set(yp)
        inter = len(s_true & s_pred)

        tp_total   += inter
        pred_total += len(s_pred)
        true_total += len(s_true)
        if s_true == s_pred:
            exact_matches += 1

    p = tp_total / pred_total if pred_total else 0.0
    r = tp_total / true_total if true_total else 0.0
    union_count = pred_total + true_total - tp_total
    j = tp_total / union_count if union_count else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = exact_matches / n_docs if n_docs else 0.0

    return (
        float(p),
        float(r),
        float(j),
        float(f1),
        float(acc),
    )
