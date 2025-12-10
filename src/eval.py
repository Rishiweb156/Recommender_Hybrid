# src/eval.py
import numpy as np

def recall_at_k(true_items, pred_items, k):
    if len(true_items) == 0:
        return 0.0
    pred_k = pred_items[:k]
    return len(set(pred_k) & set(true_items)) / len(true_items)

def precision_at_k(true_items, pred_items, k):
    pred_k = pred_items[:k]
    return len(set(pred_k) & set(true_items)) / k

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    return np.sum((2**rels - 1) / np.log2(np.arange(2, rels.size + 2)))

def ndcg_at_k(true_items, pred_items, k):
    rels = [1 if p in true_items else 0 for p in pred_items[:k]]
    dcg = dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0
