import re
from typing import Any


FIELDS = ["company", "address", "date", "total"]


def normalize_text(value: Any) -> str:
    """
    Normalize text for fair comparison.

    - Convert to string
    - Lowercase
    - Remove punctuation differences
    - Collapse whitespace
    """
    if value is None:
        return ""

    text = str(value).lower().strip()

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_total(value: Any) -> str:
    """
    Normalize monetary values.

    Examples:
        9.00 -> 9
        "9.00" -> 9
        "RM 9.00" -> 9
    """
    if value is None:
        return ""

    text = str(value).lower().strip()

    text = text.replace("rm", "")
    text = text.replace("$", "")
    text = text.replace(",", "")

    try:
        return f"{float(text):.2f}"
    except ValueError:
        return normalize_text(text)


def exact_match(prediction: Any, ground_truth: Any) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def total_match(prediction: Any, ground_truth: Any) -> bool:
    return normalize_total(prediction) == normalize_total(ground_truth)


def token_f1(prediction: Any, ground_truth: Any) -> float:
    """
    Token-level F1 score for text fields.
    """
    pred_tokens = normalize_text(prediction).split()
    true_tokens = normalize_text(ground_truth).split()

    if not pred_tokens and not true_tokens:
        return 1.0

    if not pred_tokens or not true_tokens:
        return 0.0

    pred_counts = {}
    true_counts = {}

    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    for token in true_tokens:
        true_counts[token] = true_counts.get(token, 0) + 1

    common = 0

    for token in pred_counts:
        common += min(
            pred_counts[token],
            true_counts.get(token, 0),
        )

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(true_tokens)

    return 2 * precision * recall / (precision + recall)


def evaluate_receipt(prediction, ground_truth: dict) -> dict:
    """
    Evaluate one receipt prediction against SROIE ground truth.
    """

    results = {}

    for field in FIELDS:
        predicted = getattr(prediction, field)
        expected = ground_truth[field]

        if field == "total":
            results[field] = {
                "exact_match": total_match(predicted, expected),
                "f1": 1.0 if total_match(predicted, expected) else 0.0,
                "prediction": predicted,
                "ground_truth": expected,
            }
        else:
            results[field] = {
                "exact_match": exact_match(predicted, expected),
                "f1": token_f1(predicted, expected),
                "prediction": predicted,
                "ground_truth": expected,
            }

    return results