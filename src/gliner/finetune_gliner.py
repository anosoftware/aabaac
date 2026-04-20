import gc
import json
import math
import os
import shutil
from pathlib import Path

import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from gliner import GLiNER


SET_ID = "1"

BASE_MODEL = "urchade/gliner_small-v2.1"
DATA_DIR = Path(f"../../data/set{SET_ID}")
job_id = os.environ.get("SLURM_JOB_ID", "manual")
OUTPUT_DIR = "gliner_runs"

NUM_STEPS = 500
BATCH_SIZE = 8
LR = 5e-6
WEIGHT_DECAY = 0.01
OTHERS_LR = 1e-5
OTHERS_WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear" 
WARMUP_RATIO = 0.1
GRAD_ACCUM_STEPS = 1

THRESHOLD = 0.5
SEED = 42
ALLOW_CPU = False
RESULTS_CSV = f"results_gliner_finetune_small_set{SET_ID}.csv"
BASE_PREDICTIONS_JSONL = f"predictions_base_set{SET_ID}_small.jsonl"
FINETUNED_PREDICTIONS_JSONL = f"predictions_finetuned_set{SET_ID}_small.jsonl"


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


def pick_precision() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False

    major, _minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return True, False
    return False, True


def require_gpu(allow_cpu: bool) -> None:
    if torch.cuda.is_available() or allow_cpu:
        return

    raise RuntimeError(
        "CUDA is not available in this environment. Aborting."
    )


def normalize(x: str) -> str:
    return x.lower()


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def extract_losses(output_dir: str | Path, run_id: str):
    path = Path(output_dir) / f"trainer_state{SET_ID}.json"
    if not path.exists():
        return None

    with open(path) as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    if not logs:
        return None

    rows = []
    for rec in logs:
        if "step" not in rec:
            continue
        rows.append(
            {
                "step": rec.get("step"),
                "epoch": rec.get("epoch"),
                "train_loss": rec.get("loss"),
                "eval_loss": rec.get("eval_loss"),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = (
        df.groupby("step", as_index=False)
        .agg(
            epoch=("epoch", _first_non_null),
            train_loss=("train_loss", _first_non_null),
            eval_loss=("eval_loss", _first_non_null),
        )
        .sort_values("step")
        .reset_index(drop=True)
    )

    df[["epoch", "train_loss", "eval_loss"]] = df[["epoch", "train_loss", "eval_loss"]].ffill()
    df["run"] = run_id
    df["record_type"] = "step"
    df["is_final"] = False

    final_step = state.get("global_step")
    if final_step is None or pd.isna(final_step):
        final_step = int(df["step"].max())

    final_epoch = state.get("epoch")
    if final_epoch is None or pd.isna(final_epoch):
        final_epoch = df["epoch"].iloc[-1]

    final_train_loss = df["train_loss"].iloc[-1]
    final_eval_loss = df["eval_loss"].iloc[-1]

    final_row = pd.DataFrame(
        [
            {
                "step": final_step,
                "epoch": final_epoch,
                "train_loss": final_train_loss,
                "eval_loss": final_eval_loss,
                "run": run_id,
                "record_type": "final",
                "is_final": True,
            }
        ]
    )

    return pd.concat([df, final_row], ignore_index=True)


def evaluate(model, data, labels, threshold: float):
    tp = fp = fn = 0
    per_label = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    prediction_rows = []

    for i, rec in enumerate(data):
        gold_entities = rec["gold_entities"]
        gold = {
            (e["start"], e["end"], normalize(e["label"]))
            for e in gold_entities
        }

        raw_preds = model.predict_entities(rec["text"], labels, threshold=threshold)
        pred = {
            (p["start"], p["end"], normalize(p["label"]))
            for p in raw_preds
        }

        prediction_rows.append(
            {
                "index": i,
                "pmid": rec.get("pmid"),
                "text": rec["text"],
                "gold_entities": gold_entities,
                "predicted_entities": [
                    {
                        "start": p["start"],
                        "end": p["end"],
                        "label": p["label"],
                        "text": p.get("text", rec["text"][p["start"]:p["end"]]),
                        **({"score": p["score"]} if "score" in p else {}),
                    }
                    for p in raw_preds
                ],
            }
        )

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
    return global_df, per_label_df, prediction_rows


def save_predictions_jsonl(rows, path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


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


def format_finetuning_rows(loss_df: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "row_type",
        "stage",
        "dataset",
        "scope",
        "label",
        "step",
        "epoch",
        "train_loss",
        "eval_loss",
        "record_type",
        "is_final",
    ]

    if loss_df is None or loss_df.empty:
        return pd.DataFrame(columns=columns)

    rows = loss_df.copy()
    rows["row_type"] = "finetuning"
    rows["stage"] = "finetuned"
    rows["dataset"] = "train"
    rows["scope"] = "all_labels"
    rows["label"] = "ALL"
    return rows[columns]


def safe_cleanup(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(
    base_model: str,
    train_data,
    out: str,
    batch_size: int,
    grad_accum_steps: int,
    num_steps: int,
    lr: float,
    weight_decay: float,
    others_lr: float,
    others_weight_decay: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    allow_cpu: bool,
):
    if os.path.exists(out):
        shutil.rmtree(out)

    require_gpu(allow_cpu=allow_cpu)
    use_bf16, use_fp16 = pick_precision()
    print(f"precision: bf16={use_bf16}, fp16={use_fp16}")

    model = GLiNER.from_pretrained(base_model)

    data_size = len(train_data)
    num_batches = max(1, data_size // batch_size)
    num_epochs = max(1, num_steps // num_batches)
    print(
        f"train_size={data_size} batch_size={batch_size} num_batches={num_batches} "
        f"num_steps={num_steps} num_epochs={num_epochs} grad_accum_steps={grad_accum_steps}"
    )

    train_kwargs = dict(
        train_dataset=train_data,
        eval_dataset=None,
        output_dir=out,
        max_steps=num_steps,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        others_lr=others_lr,
        others_weight_decay=others_weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=max(1, num_steps // 10),
        save_strategy="no",
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    model.train_model(**train_kwargs)
    return model



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

    train_pmids = sorted(set(c["pmid"] for c in train_data))
    test_pmids = sorted(set(c["pmid"] for c in test_data))

    print(f"Train chunks: {len(train_data)} ({len(train_pmids)} docs)")
    print(f"Test chunks: {len(test_data)} ({len(test_pmids)} docs)")
    print(f"Labels: {labels}")

    result_frames = []

    print("\n=== Baseline (zero-shot) on test set ===")
    base = GLiNER.from_pretrained(BASE_MODEL)
    base_global, base_per_label, base_predictions = evaluate(base, test_data, labels, threshold=THRESHOLD)
    print(base_global.to_string(index=False))
    print(base_per_label.to_string(index=False))
    result_frames.append(format_evaluation_rows(base_global, base_per_label, stage="base", dataset="test"))
    save_predictions_jsonl(base_predictions, output_dir / BASE_PREDICTIONS_JSONL)
    safe_cleanup(base)

    print("\n=== Fine-tuning on train set ===")
    train_out_dir = str(output_dir / "train_run")
    model = train(
        base_model=BASE_MODEL,
        train_data=train_data,
        out=train_out_dir,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        num_steps=NUM_STEPS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        others_lr=OTHERS_LR,
        others_weight_decay=OTHERS_WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        allow_cpu=ALLOW_CPU,
    )

    loss_df = extract_losses(train_out_dir, "train")
    result_frames.append(format_finetuning_rows(loss_df))
    shutil.rmtree(train_out_dir, ignore_errors=True)

    print("\n=== Fine-tuned model on test set ===")
    test_global, test_per_label, finetuned_predictions = evaluate(model, test_data, labels, threshold=THRESHOLD)
    print(test_global.to_string(index=False))
    print(test_per_label.to_string(index=False))
    result_frames.append(format_evaluation_rows(test_global, test_per_label, stage="finetuned", dataset="test"))
    save_predictions_jsonl(finetuned_predictions, output_dir / FINETUNED_PREDICTIONS_JSONL)
    safe_cleanup(model)

    results_df = pd.concat(result_frames, ignore_index=True, sort=False)
    results_path = output_dir / RESULTS_CSV
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved combined results to {results_path}")
    print(f"Saved base predictions to {output_dir / BASE_PREDICTIONS_JSONL}")
    print(f"Saved fine-tuned predictions to {output_dir / FINETUNED_PREDICTIONS_JSONL}")


if __name__ == "__main__":
    main()