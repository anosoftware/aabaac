#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from quickumls import QuickUMLS

SET_ID = "5"
TARGET_LABELS = {"Disease", "Autoantibody"}


@dataclass
class Example:
    text: str
    entities: List[Dict[str, Any]]
    id: Optional[str] = None


def load_examples_from_json(path: str) -> List[Example]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[Example] = []
    for i, row in enumerate(data):
        text = str(row.get("text", ""))
        raw_entities = row.get("gold_entities", [])

        filtered = []
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue

            label = ent.get("label")
            start = ent.get("start")
            end = ent.get("end")

            if label not in TARGET_LABELS:
                continue
            if start is None or end is None:
                continue

            try:
                start_i = int(start)
                end_i = int(end)
            except Exception:
                continue

            if not (0 <= start_i < end_i <= len(text)):
                continue

            filtered.append(
                {
                    "label": label,
                    "start": start_i,
                    "end": end_i,
                    "text": text[start_i:end_i],
                }
            )

        ex_id = str(row.get("pmid", i))
        examples.append(Example(text=text, entities=filtered, id=ex_id))

    return examples


def build_matcher(index_dir: str, threshold: float = 0.9) -> QuickUMLS:
    return QuickUMLS(
        index_dir,
        overlapping_criteria="score",
        threshold=threshold,
        similarity_name="jaccard",
        window=5,
        accepted_semtypes=None,
    )


def _flatten_quickumls_matches(
    raw_matches: List[List[Dict[str, Any]]],
    label: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, str]] = set()

    for group in raw_matches:
        for cand in group:
            start = cand.get("start")
            end = cand.get("end")
            ngram = cand.get("ngram")
            similarity = cand.get("similarity")
            cui = cand.get("cui")

            if start is None or end is None:
                continue

            try:
                start_i = int(start)
                end_i = int(end)
            except Exception:
                continue

            key = (start_i, end_i, label)
            if key in seen:
                continue
            seen.add(key)

            out.append(
                {
                    "label": label,
                    "start": start_i,
                    "end": end_i,
                    "text": ngram,
                    "cui": cui,
                    "similarity": similarity,
                }
            )

    out.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return out


def detect_entities_with_two_matchers(
    text: str,
    disease_matcher: QuickUMLS,
    autoantibody_matcher: QuickUMLS,
) -> List[Dict[str, Any]]:
    disease_raw = disease_matcher.match(text, best_match=True, ignore_syntax=False)
    auto_raw = autoantibody_matcher.match(text, best_match=True, ignore_syntax=False)

    preds: List[Dict[str, Any]] = []
    preds.extend(_flatten_quickumls_matches(disease_raw, "Disease"))
    preds.extend(_flatten_quickumls_matches(auto_raw, "Autoantibody"))

    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int, str]] = set()
    for ent in preds:
        key = (ent["start"], ent["end"], ent["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ent)

    deduped.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return deduped


def span_set(entities: Iterable[Dict[str, Any]]) -> Set[Tuple[int, int, str]]:
    return {(int(e["start"]), int(e["end"]), str(e["label"])) for e in entities}


def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


def evaluate(
    examples: Sequence[Example],
    disease_matcher: QuickUMLS,
    autoantibody_matcher: QuickUMLS,
) -> Dict[str, Any]:
    tp = fp = fn = 0
    all_predictions: List[Dict[str, Any]] = []

    per_label_tp = defaultdict(int)
    per_label_fp = defaultdict(int)
    per_label_fn = defaultdict(int)

    for ex in examples:
        pred_entities = detect_entities_with_two_matchers(
            ex.text,
            disease_matcher=disease_matcher,
            autoantibody_matcher=autoantibody_matcher,
        )

        gold = span_set(ex.entities)
        pred = span_set(pred_entities)

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

        for label in TARGET_LABELS:
            gold_l = {(s, e, l) for (s, e, l) in gold if l == label}
            pred_l = {(s, e, l) for (s, e, l) in pred if l == label}

            per_label_tp[label] += len(gold_l & pred_l)
            per_label_fp[label] += len(pred_l - gold_l)
            per_label_fn[label] += len(gold_l - pred_l)

        all_predictions.append(
            {
                "id": ex.id,
                "text": ex.text,
                "gold_entities": ex.entities,
                "predicted_entities": pred_entities,
            }
        )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    label_metrics = []
    for label in sorted(TARGET_LABELS):
        p = safe_div(per_label_tp[label], per_label_tp[label] + per_label_fp[label])
        r = safe_div(per_label_tp[label], per_label_tp[label] + per_label_fn[label])
        f = safe_div(2 * p * r, p + r)
        label_metrics.append(
            {
                "label": label,
                "tp": per_label_tp[label],
                "fp": per_label_fp[label],
                "fn": per_label_fn[label],
                "precision": round(p, 6),
                "recall": round(r, 6),
                "f1": round(f, 6),
            }
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "n_examples": len(examples),
        "predictions": all_predictions,
        "label_metrics": label_metrics,
    }


def write_results_csv(results: Dict[str, Any], output_path: str) -> None:
    rows = [
        {
            "label": "ALL",
            "tp": results["tp"],
            "fp": results["fp"],
            "fn": results["fn"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
        }
    ]
    rows.extend(results["label_metrics"])

    fieldnames = ["label", "tp", "fp", "fn", "precision", "recall", "f1"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    TEST_JSON = f"../data/../set{SET_ID}/test.json"

    DISEASE_QUICKUMLS_DIR = "quickumls_disease"
    AUTOANTIBODY_QUICKUMLS_DIR = "quickumls_aab"

    disease_matcher = build_matcher(DISEASE_QUICKUMLS_DIR, threshold=0.9)
    autoantibody_matcher = build_matcher(AUTOANTIBODY_QUICKUMLS_DIR, threshold=0.9)

    test_examples = load_examples_from_json(TEST_JSON)
    results = evaluate(
        test_examples,
        disease_matcher=disease_matcher,
        autoantibody_matcher=autoantibody_matcher,
    )

    print(
        json.dumps(
            {
                "tp": results["tp"],
                "fp": results["fp"],
                "fn": results["fn"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1": results["f1"],
                "n_examples": results["n_examples"],
                "label_metrics": results["label_metrics"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    with open(
        f"result_quickumls_set{SET_ID}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results["predictions"], f, ensure_ascii=False, indent=2)

    write_results_csv(
        results,
        f"result_quickumls_set{SET_ID}_metrics.csv",
    )