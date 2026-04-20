import gc
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from gliner import GLiNER


n = "1"


BASE_MODEL = "urchade/gliner_large-v2.1"
DATA_DIR = Path(f"../../data/set{n}")
job_id = os.environ.get("SLURM_JOB_ID", "manual")
OUTPUT_DIR = "gliner_runs"

THRESHOLD = 0.3
SEED = 42
ALLOW_CPU = True
MAX_LEN = 512
RESULTS_CSV = f"results{n}_few_large.csv"
PREDICTIONS_JSONL = f"predictions{n}_few_large.jsonl"

FEWSHOT_EXAMPLES = [
    {
        "sentence": "The patient had elevated titers of AQP4 antibodies.",
        "entities": [
            ("Autoantibody", "AQP4 antibodies"),
        ],
    },
    {
        "sentence": "Anti-nuclear antibodies were found in the blood sample.",
        "entities": [
            ("Autoantibody", "Anti-nuclear antibodies"),
        ],
    },
    {
        "sentence": "Autoantibodies were detected in the serum of the patient.",
        "entities": [
            ("Autoantibody location", "serum"),
        ],
    },
    {
        "sentence": "The antibodies were identified in cerebrospinal fluid.",
        "entities": [
            ("Autoantibody location", "cerebrospinal fluid"),
        ],
    },
    {
        "sentence": "Thyroperoxidase antibodies were present in the sample.",
        "entities": [
            ("Autoantibody target", "Thyroperoxidase"),
        ],
    },
    {
        "sentence": "Anti-GBM antibodies target the glomerular basement membrane.",
        "entities": [
            ("Autoantibody target", "glomerular basement membrane"),
        ],
    },
    {
        "sentence": "The patient was diagnosed with diabetes type 1.",
        "entities": [
            ("Disease", "diabetes type 1"),
        ],
    },
    {
        "sentence": "She developed Goodpasture syndrome after the onset of symptoms.",
        "entities": [
            ("Disease", "Goodpasture syndrome"),
        ],
    },
    {
        "sentence": "The patient was admitted after complaining of thigh pain.",
        "entities": [
            ("Symptom or clinical sign", "thigh pain"),
        ],
    },
    {
        "sentence": "The patient presented with pulmonary hemorrhage.",
        "entities": [
            ("Symptom or clinical sign", "pulmonary hemorrhage"),
        ],
    },
]
# -----------------------------------------------------------------


def log_cuda_status() -> dict:
    info = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    print(json.dumps(info, indent=2))

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        gpu_info = {
            "gpu_name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "capability": f"{capability[0]}.{capability[1]}",
        }
        print(json.dumps(gpu_info, indent=2))

    return info


def require_gpu(allow_cpu: bool) -> None:
    if torch.cuda.is_available() or allow_cpu:
        return
    raise RuntimeError("CUDA is not available in this environment. Aborting.")


def normalize(x: str) -> str:
    return x.lower().strip()


def safe_cleanup(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_fewshot_prefix(examples: list[dict]) -> str:
    parts: list[str] = []
    for ex in examples:
        parts.append(f'"sentence": "{ex["sentence"]}"')
        for label, mention in ex["entities"]:
            parts.append(f'"{label}": "{mention}"')
        parts.append("")
    parts.append('"sentence": "')
    return "\n".join(parts)


def tokenize_record_to_text(record: dict) -> tuple[str, list[tuple[int, int, str]]]:
    if "text" in record and "gold_entities" in record:
        text = record["text"]
        gold = [(e["start"], e["end"], e["label"]) for e in record["gold_entities"]]
        return text, gold

    if "tokenized_text" in record and "ner" in record:
        tokens = record["tokenized_text"]
        token_starts: list[int] = []
        token_ends: list[int] = []
        text_parts: list[str] = []
        char_pos = 0

        for i, tok in enumerate(tokens):
            if i > 0:
                text_parts.append(" ")
                char_pos += 1
            token_starts.append(char_pos)
            text_parts.append(tok)
            char_pos += len(tok)
            token_ends.append(char_pos)

        text = "".join(text_parts)
        gold: list[tuple[int, int, str]] = []
        for start_tok, end_tok, label in record["ner"]:
            start_char = token_starts[start_tok]
            end_char = token_ends[end_tok]
            gold.append((start_char, end_char, label))
        return text, gold

    raise KeyError("Record must contain either (text, gold_entities) or (tokenized_text, ner).")


def evaluate(model, data, labels, threshold: float, fewshot_prefix: str):
    tp = fp = fn = 0
    per_label = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    prediction_dump: list[dict] = []

    prefix_len = len(fewshot_prefix)
    suffix = '"'

    for idx, rec in enumerate(data):
        text, gold_entities = tokenize_record_to_text(rec)

        gold = {
            (start, end, normalize(label))
            for start, end, label in gold_entities
        }

        full_text = fewshot_prefix + text + suffix
        raw_preds = model.predict_entities(full_text, labels, threshold=threshold)

        target_start = prefix_len
        target_end = prefix_len + len(text)
        pred = set()
        filtered_preds: list[dict] = []
        for p in raw_preds:
            start = p["start"]
            end = p["end"]
            if start < target_start or end > target_end:
                continue
            shifted = {
                **p,
                "start": start - prefix_len,
                "end": end - prefix_len,
            }
            filtered_preds.append(shifted)
            pred.add((
                shifted["start"],
                shifted["end"],
                normalize(shifted["label"]),
            ))

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

        for label in labels:
            l = normalize(label)
            gold_l = {x for x in gold if x[2] == l}
            pred_l = {x for x in pred if x[2] == l}
            per_label[label]["tp"] += len(gold_l & pred_l)
            per_label[label]["fp"] += len(pred_l - gold_l)
            per_label[label]["fn"] += len(gold_l - pred_l)

        prediction_dump.append({
            "index": idx,
            "pmid": rec.get("pmid"),
            "text": text,
            "gold_entities": [
                {"start": start, "end": end, "label": label, "text": text[start:end]}
                for start, end, label in gold_entities
            ],
            "predicted_entities": filtered_preds,
        })

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    global_df = pd.DataFrame([{
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }])

    rows = []
    for label, c in per_label.items():
        tp_l, fp_l, fn_l = c["tp"], c["fp"], c["fn"]
        p_l = tp_l / (tp_l + fp_l) if tp_l + fp_l else 0.0
        r_l = tp_l / (tp_l + fn_l) if tp_l + fn_l else 0.0
        f1_l = 2 * p_l * r_l / (p_l + r_l) if p_l + r_l else 0.0
        rows.append({
            "label": label,
            "precision": p_l,
            "recall": r_l,
            "f1": f1_l,
            "tp": tp_l,
            "fp": fp_l,
            "fn": fn_l,
        })

    per_label_df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    return global_df, per_label_df, prediction_dump


def format_evaluation_rows(global_df: pd.DataFrame, per_label_df: pd.DataFrame, stage: str, dataset: str) -> pd.DataFrame:
    global_rows = global_df.copy()
    global_rows["stage"] = stage
    global_rows["dataset"] = dataset
    global_rows["scope"] = "all_labels"
    global_rows["label"] = "ALL"
    global_rows["row_type"] = "evaluation"

    per_label_rows = per_label_df.copy()
    per_label_rows["stage"] = stage
    per_label_rows["dataset"] = dataset
    per_label_rows["scope"] = "per_label"
    per_label_rows["row_type"] = "evaluation"

    cols = [
        "row_type",
        "stage",
        "dataset",
        "scope",
        "label",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
    ]
    return pd.concat([global_rows[cols], per_label_rows[cols]], ignore_index=True)



def main() -> None:
    log_cuda_status()
    require_gpu(allow_cpu=ALLOW_CPU)

    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "train.json") as f:
        train_data = json.load(f)
    with open(data_dir / "test.json") as f:
        test_data = json.load(f)
    with open(data_dir / "labels.json") as f:
        labels = json.load(f)

    train_pmids = sorted(set(c.get("pmid") for c in train_data if c.get("pmid") is not None))
    test_pmids = sorted(set(c.get("pmid") for c in test_data if c.get("pmid") is not None))

    print(f"Train chunks: {len(train_data)} ({len(train_pmids)} docs)")
    print(f"Test chunks: {len(test_data)} ({len(test_pmids)} docs)")
    print(f"Labels: {labels}")

    fewshot_prefix = build_fewshot_prefix(FEWSHOT_EXAMPLES)
    print("\n=== Few-shot prefix ===")
    print(fewshot_prefix)

    result_frames = []

    print("\n=== Baseline (few-shot text prefix) on test set ===")
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    base = GLiNER.from_pretrained(BASE_MODEL, map_location=map_location)
    if hasattr(base, "config") and hasattr(base.config, "max_len"):
        print(f"Original max_len: {base.config.max_len}")
        base.config.max_len = max(int(base.config.max_len), MAX_LEN)
        print(f"Updated max_len: {base.config.max_len}")
    base_global, base_per_label, prediction_dump = evaluate(
        base,
        test_data,
        labels,
        threshold=THRESHOLD,
        fewshot_prefix=fewshot_prefix,
    )
    print(base_global.to_string(index=False))
    print(base_per_label.to_string(index=False))
    result_frames.append(format_evaluation_rows(base_global, base_per_label, stage="base_fewshot_text", dataset="test"))

    predictions_path = output_dir / PREDICTIONS_JSONL
    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in prediction_dump:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {predictions_path}")

    safe_cleanup(base)

    results_df = pd.concat(result_frames, ignore_index=True, sort=False)
    results_path = output_dir / RESULTS_CSV
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved combined results to {results_path}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    main()