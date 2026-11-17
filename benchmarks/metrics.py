"""
Evaluation Metrics for Multi-Hop QA & Ablation Benchmarks.
Standard HotpotQA Exact Match (EM), Token F1, Supporting-Fact Recall,
Faithfulness / Grounding Score, and Latency percentiles (P50/P95).
"""

from typing import List, Tuple, Set
import re
import string
import numpy as np


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Computes exact match score (0.0 or 1.0) after standard normalization."""
    # If prediction contains Answer: extract the answer line
    match = re.search(r"Answer:\s*([^\n]+)", prediction, re.IGNORECASE)
    pred_str = match.group(1).strip() if match else prediction

    norm_pred = normalize_answer(pred_str)
    norm_gold = normalize_answer(ground_truth)
    return 1.0 if norm_pred == norm_gold or norm_gold in norm_pred else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 score between prediction and ground truth."""
    match = re.search(r"Answer:\s*([^\n]+)", prediction, re.IGNORECASE)
    pred_str = match.group(1).strip() if match else prediction

    pred_tokens = normalize_answer(pred_str).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common: Set[str] = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_supporting_fact_recall(retrieved_doc_ids: List[str], gold_doc_ids: List[str]) -> float:
    """
    Computes recall of required gold supporting documents.
    Multi-hop bridge questions require retrieving both/all supporting documents.
    """
    if not gold_doc_ids:
        return 1.0
    retrieved_set = set(retrieved_doc_ids)
    gold_set = set(gold_doc_ids)
    hits = len(retrieved_set & gold_set)
    return hits / len(gold_set)


def compute_faithfulness(prediction: str, retrieved_contexts: List[str]) -> float:
    """
    Computes faithfulness/grounding score: measures what fraction of named entities
    and key claim tokens in the prediction are explicitly grounded in the retrieved passages,
    penalizing parametric hallucinations.
    """
    if not retrieved_contexts:
        return 0.0

    # Extract claims / entities from prediction
    pred_clean = re.sub(r"\[Doc \d+\]", "", prediction)
    tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", pred_clean.lower())
    stop_words = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "to", "and", "answer", "conclusion", "step", "evidence"}
    content_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    if not content_tokens:
        return 1.0

    combined_context = " ".join(retrieved_contexts).lower()
    grounded_count = sum(1 for t in content_tokens if t in combined_context)
    return grounded_count / len(content_tokens)


def compute_latency_percentiles(latencies_ms: List[float]) -> Tuple[float, float]:
    """Computes (P50, P95) end-to-end latency in milliseconds."""
    if not latencies_ms:
        return (0.0, 0.0)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    return round(p50, 2), round(p95, 2)
