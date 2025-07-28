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

def compute_toptal_metrics(y_true_list, y_pred_list, entity_type):
    """
    Compute Toptal-specific metrics according to adjusted task requirements:

    - For persons and locations: Only multi-word entity matching (precision and recall)
    - For organizations:
        - Precision uses only multi-word predicted entities
        - Recall considers all gold entities (multi-word + single-word)

    Returns:
        Dictionary with toptal metrics
    """
    def normalize_entity(entity):
        return ' '.join(entity.strip().split()).lower()

    def get_multi_word_entities(entities):
        return [e for e in entities if ' ' in e.strip()]

    def get_single_words(entities):
        words = []
        for entity in entities:
            words.extend(entity.strip().split())
        return list(set(words))

    multi_word_tp = multi_word_pred = multi_word_true = 0
    multi_word_exact_matches = 0

    single_word_tp = single_word_pred = single_word_true = 0
    single_word_exact_matches = 0

    n_docs = len(y_true_list)

    for yt, yp in zip(y_true_list, y_pred_list):
        yt_norm = [normalize_entity(e) for e in yt]
        yp_norm = [normalize_entity(e) for e in yp]

        # Multi-word precision: only multi-word predictions
        yp_multi = get_multi_word_entities(yp_norm)

        # Multi-word recall:
        if entity_type == "organization":
            # Count all gold entities (even single-word) for recall
            yt_multi = yt_norm
        else:
            yt_multi = get_multi_word_entities(yt_norm)

        s_true_multi, s_pred_multi = set(yt_multi), set(yp_multi)
        inter_multi = len(s_true_multi & s_pred_multi)

        multi_word_tp += inter_multi
        multi_word_pred += len(s_pred_multi)
        multi_word_true += len(s_true_multi)
        if s_true_multi == s_pred_multi:
            multi_word_exact_matches += 1

        # Single-word metrics (only for organizations)
        if entity_type == "organization":
            yt_words = get_single_words(yt_norm)
            yp_words = get_single_words(yp_norm)

            s_true_words, s_pred_words = set(yt_words), set(yp_words)
            inter_words = len(s_true_words & s_pred_words)

            single_word_tp += inter_words
            single_word_pred += len(s_pred_words)
            single_word_true += len(s_true_words)
            if s_true_words == s_pred_words:
                single_word_exact_matches += 1

    # Multi-word metrics
    multi_word_p = multi_word_tp / multi_word_pred if multi_word_pred else 0.0
    multi_word_r = multi_word_tp / multi_word_true if multi_word_true else 0.0
    multi_word_j = (
        multi_word_tp / (multi_word_pred + multi_word_true - multi_word_tp)
        if (multi_word_pred + multi_word_true - multi_word_tp) > 0 else 0.0
    )
    multi_word_f1 = (
        2 * multi_word_p * multi_word_r / (multi_word_p + multi_word_r)
        if (multi_word_p + multi_word_r) > 0 else 0.0
    )
    multi_word_acc = multi_word_exact_matches / n_docs if n_docs else 0.0

    result = {
        "multi_word": {
            "precision": round(multi_word_p, 6),
            "recall": round(multi_word_r, 6),
            "jaccard": round(multi_word_j, 6),
            "f1": round(multi_word_f1, 6),
            "accuracy": round(multi_word_acc, 6),
        }
    }

    # Single-word metrics for organizations
    if entity_type == "organization":
        single_word_p = single_word_tp / single_word_pred if single_word_pred else 0.0
        single_word_r = single_word_tp / single_word_true if single_word_true else 0.0
        single_word_j = (
            single_word_tp / (single_word_pred + single_word_true - single_word_tp)
            if (single_word_pred + single_word_true - single_word_tp) > 0 else 0.0
        )
        single_word_f1 = (
            2 * single_word_p * single_word_r / (single_word_p + single_word_r)
            if (single_word_p + single_word_r) > 0 else 0.0
        )
        single_word_acc = single_word_exact_matches / n_docs if n_docs else 0.0

        result["single_word"] = {
            "precision": round(single_word_p, 6),
            "recall": round(single_word_r, 6),
            "jaccard": round(single_word_j, 6),
            "f1": round(single_word_f1, 6),
            "accuracy": round(single_word_acc, 6),
        }

    return result
