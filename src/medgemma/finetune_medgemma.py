#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


SET_ID = "1"

# ============================================================================
# CONFIG
# ============================================================================
TRAIN_PATH = f"../../data/set{SET_ID}/train.json"
TEST_PATH = f"../../data/set{SET_ID}/test.json"
MODEL_NAME = "google/medgemma-27b-text-it"
OUTPUT_DIR = "medgemma_finetune/"
RESULTS_CSV = os.path.join(OUTPUT_DIR, f"results_medgemma_finetune_{SET_ID}.csv")

MAX_SEQ_LEN = None
MAX_NEW_TOKENS = 1024
MODEL_MAX_OUTPUT_TOKENS = 1024
REPETITION_PENALTY = 1.05
NUM_TRAIN_EPOCHS = 3.0
LEARNING_RATE = 5e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3
LR_SCHEDULER_TYPE = "constant"

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = "all-linear"

SEED = 42
USE_4BIT = True
LIMIT_TRAIN = None
LIMIT_TEST = None
SAVE_PREDICTIONS = True
STRIP_ENTITY_WHITESPACE = True
DEBUG_ALIGNMENT = False
# ============================================================================

ALL_LABELS = [
    "Autoantibody",
    "Autoantibody location",
    "Autoantibody target",
    "Disease",
    "Symptom or clinical sign",
]


LOGGER = logging.getLogger("medgemma_ner")


@dataclass
class Config:
    train_path: str = TRAIN_PATH
    test_path: str = TEST_PATH
    model_name: str = MODEL_NAME
    output_dir: str = OUTPUT_DIR
    results_csv: str = RESULTS_CSV
    max_seq_len: Optional[int] = MAX_SEQ_LEN
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS
    model_max_output_tokens: int = MODEL_MAX_OUTPUT_TOKENS
    repetition_penalty: float = REPETITION_PENALTY
    num_train_epochs: float = NUM_TRAIN_EPOCHS
    learning_rate: float = LEARNING_RATE
    per_device_train_batch_size: int = PER_DEVICE_TRAIN_BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    warmup_ratio: float = WARMUP_RATIO
    weight_decay: float = WEIGHT_DECAY
    max_grad_norm: float = MAX_GRAD_NORM
    lr_scheduler_type: str = LR_SCHEDULER_TYPE
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    target_modules: str = TARGET_MODULES
    seed: int = SEED
    use_4bit: bool = USE_4BIT
    limit_train: Optional[int] = LIMIT_TRAIN
    limit_test: Optional[int] = LIMIT_TEST
    save_predictions: bool = SAVE_PREDICTIONS
    strip_entity_whitespace: bool = STRIP_ENTITY_WHITESPACE
    debug_alignment: bool = DEBUG_ALIGNMENT


@dataclass
class Example:
    text: str
    entities: List[Dict[str, Any]]
    id: Optional[str] = None


@dataclass
class PredictedEntityText:
    label: str
    text: str
    hint_start: Optional[int] = None
    hint_end: Optional[int] = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path_obj, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
        return rows

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _trim_span_whitespace(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _clean_entities(
    entities: List[Dict[str, Any]],
    text: str,
    strip_entity_whitespace: bool,
) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        label = ent.get("label")
        if start is None or end is None or label is None:
            continue

        try:
            start_i = int(start)
            end_i = int(end)
        except Exception:
            continue

        if start_i < 0 or end_i <= start_i or end_i > len(text):
            continue

        if strip_entity_whitespace:
            start_i, end_i = _trim_span_whitespace(text, start_i, end_i)
            if end_i <= start_i:
                continue

        key = (start_i, end_i, str(label))
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(
            {
                "start": start_i,
                "end": end_i,
                "label": str(label),
                "text": text[start_i:end_i],
            }
        )

    cleaned.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return cleaned


def _load_entities_from_gold_entities(
    row: Dict[str, Any],
    text: str,
    strip_entity_whitespace: bool,
) -> List[Dict[str, Any]]:
    raw_entities = row.get("gold_entities")
    if not isinstance(raw_entities, list):
        return []

    entities: List[Dict[str, Any]] = []
    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        entities.append(
            {
                "start": ent.get("start"),
                "end": ent.get("end"),
                "label": ent.get("label"),
            }
        )
    return _clean_entities(entities, text, strip_entity_whitespace=strip_entity_whitespace)


def _token_offsets_from_text(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    cursor = 0

    for idx, token in enumerate(tokens):
        if not isinstance(token, str) or token == "":
            raise ValueError(f"Invalid token at position {idx}")

        start = text.find(token, cursor)
        if start == -1:
            start = text.find(token)
        if start == -1:
            raise ValueError(f"Could not align token {idx}={token!r} back to text")

        end = start + len(token)
        offsets.append((start, end))
        cursor = end

    return offsets


def _load_entities_from_token_ner(
    row: Dict[str, Any],
    text: str,
    strip_entity_whitespace: bool,
) -> List[Dict[str, Any]]:
    raw_ner = row.get("ner")
    tokens = row.get("tokenized_text")
    if not isinstance(raw_ner, list) or not isinstance(tokens, list):
        return []

    offsets = _token_offsets_from_text(text, tokens)
    entities: List[Dict[str, Any]] = []

    for ent in raw_ner:
        if not isinstance(ent, (list, tuple)) or len(ent) != 3:
            continue
        token_start, token_end, label = ent
        try:
            token_start_i = int(token_start)
            token_end_i = int(token_end)
        except Exception:
            continue
        if label is None:
            continue
        if token_start_i < 0 or token_end_i < token_start_i or token_end_i >= len(offsets):
            continue

        char_start = offsets[token_start_i][0]
        char_end = offsets[token_end_i][1]
        entities.append(
            {
                "start": char_start,
                "end": char_end,
                "label": str(label),
            }
        )

    return _clean_entities(entities, text, strip_entity_whitespace=strip_entity_whitespace)


def load_examples(path: str, limit: Optional[int], strip_entity_whitespace: bool) -> List[Example]:
    raw_rows = _read_json_or_jsonl(path)
    examples: List[Example] = []

    for i, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} in {path} is not a JSON object")

        text = row.get("text") or row.get("sentence") or row.get("content")
        if not isinstance(text, str):
            raise ValueError(f"Row {i} in {path} is missing a string `text` field")

        entities = _load_entities_from_gold_entities(
            row,
            text,
            strip_entity_whitespace=strip_entity_whitespace,
        )
        if not entities:
            entities = _load_entities_from_token_ner(
                row,
                text,
                strip_entity_whitespace=strip_entity_whitespace,
            )

        example_id = row.get("pmid", row.get("id", str(i)))
        examples.append(Example(text=text, entities=entities, id=str(example_id)))

    if limit is not None:
        examples = examples[:limit]
    return examples


def get_label_inventory() -> List[str]:
    labels = list(dict.fromkeys(ALL_LABELS))
    if not labels:
        raise ValueError("ALL_LABELS is empty")
    return labels


def validate_examples_labels(examples: Sequence[Example], split_name: str) -> None:
    allowed = set(get_label_inventory())
    unexpected_labels = sorted(
        {
            str(ent["label"]).strip()
            for ex in examples
            for ent in ex.entities
            if str(ent["label"]).strip() not in allowed
        }
    )
    if unexpected_labels:
        raise ValueError(
            f"Found labels in {split_name} data that are missing from ALL_LABELS: "
            + ", ".join(unexpected_labels)
        )


def system_prompt(labels: Sequence[str]) -> str:
    label_text = ", ".join(labels)
    return (
        "You perform biomedical named entity recognition on the user's text. "
        "Return only compact valid JSON with this exact schema: "
        '{"entities":[{"label":"<allowed_label>","text":"<exact substring from the passage>"}]}. '
        "Do not return character offsets. Do not add explanations. Do not add markdown. "
        "Copy each entity text exactly from the passage. "
        "If the same mention appears multiple times in the passage, repeat it once per occurrence. "
        f"Allowed labels: {label_text}. "
        "If there are no entities, return {\"entities\":[]}."
    )


def user_prompt(text: str) -> str:
    return (
        "Extract all entities from the passage below.\n\n"
        "Passage:\n"
        f"{text}\n\n"
        "Return JSON only."
    )


def gold_response_json(example: Example) -> str:
    payload = {
        "entities": [
            {
                "label": ent["label"],
                "text": example.text[ent["start"]:ent["end"]],
            }
            for ent in example.entities
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_training_prompt(tokenizer: AutoTokenizer, example: Example, labels: Sequence[str]) -> str:
    messages = [
        {"role": "system", "content": system_prompt(labels)},
        {"role": "user", "content": user_prompt(example.text)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def render_training_completion(tokenizer: AutoTokenizer, example: Example) -> str:
    eos = tokenizer.eos_token or ""
    return gold_response_json(example) + eos


def build_eval_messages(example: Example, labels: Sequence[str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(labels)},
        {"role": "user", "content": user_prompt(example.text)},
    ]


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_balanced_json(text: str) -> Optional[str]:
    text = strip_code_fences(text)
    start_positions = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in start_positions:
        opening = text[start]
        closing = "]" if opening == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _normalize_label(label: Any) -> str:
    return str(label).strip()


def _normalize_surface_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _all_exact_occurrences(haystack: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + 1
    return spans


def _surface_to_regex(surface: str) -> Optional[re.Pattern[str]]:
    cleaned = _normalize_surface_text(surface)
    if not cleaned:
        return None

    pieces = re.split(r"(\s+)", cleaned)
    pattern_parts: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        if piece.isspace():
            pattern_parts.append(r"\s+")
            continue
        piece = re.escape(piece)
        piece = piece.replace(r"\-", r"[-‐‑–—]")
        pattern_parts.append(piece)

    if not pattern_parts:
        return None
    return re.compile("".join(pattern_parts), flags=re.IGNORECASE)


def _all_regex_occurrences(haystack: str, surface: str) -> List[Tuple[int, int]]:
    pattern = _surface_to_regex(surface)
    if pattern is None:
        return []
    return [(m.start(), m.end()) for m in pattern.finditer(haystack)]


def _dedupe_spans(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for span in spans:
        if span in seen:
            continue
        seen.add(span)
        out.append(span)
    out.sort()
    return out


def _candidate_spans_for_entity(source_text: str, ent_text: str) -> List[Tuple[int, int]]:
    variants: List[str] = []
    raw = ent_text
    stripped = ent_text.strip()
    normalized = _normalize_surface_text(ent_text)

    for candidate in (raw, stripped, normalized):
        if candidate and candidate not in variants:
            variants.append(candidate)

    spans: List[Tuple[int, int]] = []
    for variant in variants:
        spans.extend(_all_exact_occurrences(source_text, variant))
    if not spans:
        for variant in variants:
            spans.extend(_all_regex_occurrences(source_text, variant))
    return _dedupe_spans(spans)


def normalize_predicted_entities(
    data: Any,
    source_text: str,
    allowed_labels: Sequence[str],
) -> List[PredictedEntityText]:
    if isinstance(data, dict):
        entities = data.get("entities", [])
    elif isinstance(data, list):
        entities = data
    else:
        entities = []

    allowed = set(allowed_labels)
    normalized: List[PredictedEntityText] = []

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        label = ent.get("label", ent.get("type", ent.get("entity")))
        if label is None:
            continue
        label_s = _normalize_label(label)
        if label_s not in allowed:
            continue

        text_value = ent.get("text")
        hint_start = ent.get("start", ent.get("begin", ent.get("start_offset")))
        hint_end = ent.get("end", ent.get("stop", ent.get("end_offset")))

        start_i: Optional[int] = None
        end_i: Optional[int] = None
        if hint_start is not None and hint_end is not None:
            try:
                start_i = int(hint_start)
                end_i = int(hint_end)
            except Exception:
                start_i = None
                end_i = None

        if not isinstance(text_value, str) or not text_value.strip():
            if start_i is None or end_i is None:
                continue
            if start_i < 0 or end_i <= start_i or end_i > len(source_text):
                continue
            text_value = source_text[start_i:end_i]

        normalized.append(
            PredictedEntityText(
                label=label_s,
                text=str(text_value),
                hint_start=start_i,
                hint_end=end_i,
            )
        )

    return normalized


def parse_model_output(
    raw_text: str,
    source_text: str,
    allowed_labels: Sequence[str],
) -> List[PredictedEntityText]:
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    candidates = [raw_text]
    extracted = extract_balanced_json(raw_text)
    if extracted and extracted != raw_text:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            data = json.loads(strip_code_fences(candidate))
            return normalize_predicted_entities(data, source_text, allowed_labels)
        except Exception:
            continue
    return []


def align_predicted_entities_to_spans(
    predicted: Sequence[PredictedEntityText],
    source_text: str,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    used_spans: set[Tuple[int, int, str]] = set()
    aligned: List[Dict[str, Any]] = []

    for ent in predicted:
        candidate_spans = _candidate_spans_for_entity(source_text, ent.text)

        if ent.hint_start is not None and ent.hint_end is not None:
            hinted = (ent.hint_start, ent.hint_end)
            if 0 <= hinted[0] < hinted[1] <= len(source_text):
                hinted_text = source_text[hinted[0]:hinted[1]]
                if _normalize_surface_text(hinted_text) == _normalize_surface_text(ent.text):
                    candidate_spans = [hinted] + [span for span in candidate_spans if span != hinted]

        chosen: Optional[Tuple[int, int]] = None
        for span in candidate_spans:
            key = (span[0], span[1], ent.label)
            if key not in used_spans:
                chosen = span
                break

        if chosen is None:
            if debug:
                LOGGER.debug("Could not align predicted entity label=%s text=%r", ent.label, ent.text)
            continue

        key = (chosen[0], chosen[1], ent.label)
        used_spans.add(key)
        aligned.append(
            {
                "start": chosen[0],
                "end": chosen[1],
                "label": ent.label,
                "text": source_text[chosen[0]:chosen[1]],
            }
        )

    aligned.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return aligned


def load_model_and_tokenizer(model_name: str, use_4bit: bool) -> Tuple[Any, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("USE_4BIT=True requires CUDA")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )

    model_kwargs = dict(
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = True
    return model, tokenizer


def _get_generation_device(model: Any) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def generate_prediction(
    model: Any,
    tokenizer: AutoTokenizer,
    example: Example,
    labels: Sequence[str],
    max_new_tokens: int,
    repetition_penalty: float,
    debug_alignment: bool,
) -> Tuple[List[Dict[str, Any]], str, List[PredictedEntityText]]:
    messages = build_eval_messages(example, labels)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = _get_generation_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            renormalize_logits=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generation[0][prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    parsed_text_entities = parse_model_output(decoded, example.text, labels)
    aligned_entities = align_predicted_entities_to_spans(
        parsed_text_entities,
        example.text,
        debug=debug_alignment,
    )
    return aligned_entities, decoded, parsed_text_entities


def span_set(entities: Iterable[Dict[str, Any]]) -> set[Tuple[int, int, str]]:
    return {(int(e["start"]), int(e["end"]), str(e["label"])) for e in entities}


def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


def compute_metrics(
    gold_examples: Sequence[Example],
    predicted_entities_per_example: Sequence[List[Dict[str, Any]]],
    labels: Sequence[str],
    stage: str,
    model_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    all_gold = [span_set(ex.entities) for ex in gold_examples]
    all_pred = [span_set(preds) for preds in predicted_entities_per_example]

    def counts_for_label(label: Optional[str]) -> Tuple[int, int, int, int, int]:
        tp = fp = fn = gold_count = pred_count = 0
        for gold_spans, pred_spans in zip(all_gold, all_pred):
            if label is None:
                gold_f = gold_spans
                pred_f = pred_spans
            else:
                gold_f = {x for x in gold_spans if x[2] == label}
                pred_f = {x for x in pred_spans if x[2] == label}
            tp += len(gold_f & pred_f)
            fp += len(pred_f - gold_f)
            fn += len(gold_f - pred_f)
            gold_count += len(gold_f)
            pred_count += len(pred_f)
        return tp, fp, fn, gold_count, pred_count

    for label in [None, *labels]:
        tp, fp, fn, gold_count, pred_count = counts_for_label(label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "stage": stage,
                "label": "__ALL__" if label is None else label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gold_count": gold_count,
                "pred_count": pred_count,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "n_test_examples": len(gold_examples),
                "model_name": model_name,
            }
        )

    return rows


def evaluate_model(
    model: Any,
    tokenizer: AutoTokenizer,
    test_examples: Sequence[Example],
    labels: Sequence[str],
    max_new_tokens: int,
    repetition_penalty: float,
    stage: str,
    model_name: str,
    debug_alignment: bool,
    save_predictions_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    predicted_all: List[List[Dict[str, Any]]] = []
    prediction_records: List[Dict[str, Any]] = []

    for example in tqdm(test_examples, desc=f"Evaluating {stage}"):
        preds, raw_output, parsed_text_entities = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            example=example,
            labels=labels,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            debug_alignment=debug_alignment,
        )
        predicted_all.append(preds)
        prediction_records.append(
            {
                "id": example.id,
                "text": example.text,
                "gold_entities": example.entities,
                "parsed_text_entities": [
                    {
                        "label": ent.label,
                        "text": ent.text,
                        "hint_start": ent.hint_start,
                        "hint_end": ent.hint_end,
                    }
                    for ent in parsed_text_entities
                ],
                "predicted_entities": preds,
                "raw_model_output": raw_output,
                "stage": stage,
            }
        )

    if save_predictions_path is not None:
        with open(save_predictions_path, "w", encoding="utf-8") as f:
            for row in prediction_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_metrics(
        gold_examples=test_examples,
        predicted_entities_per_example=predicted_all,
        labels=labels,
        stage=stage,
        model_name=model_name,
    )


def build_train_dataset(
    tokenizer: AutoTokenizer,
    train_examples: Sequence[Example],
    labels: Sequence[str],
    max_seq_len: Optional[int],
) -> Dataset:
    rows: List[Dict[str, str]] = []
    lengths: List[int] = []

    for ex in train_examples:
        prompt = render_training_prompt(tokenizer, ex, labels)
        completion = render_training_completion(tokenizer, ex)
        total_len = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
        lengths.append(total_len)
        rows.append({"prompt": prompt, "completion": completion})

    if not rows:
        raise ValueError("No training examples were loaded.")

    LOGGER.info(
        "Prepared %d training examples with no manual length filtering. Tokenized length stats: min=%d median=%d max=%d | trainer max_length=%s",
        len(rows),
        min(lengths),
        sorted(lengths)[len(lengths) // 2],
        max(lengths),
        max_seq_len,
    )
    return Dataset.from_list(rows)


def build_peft_config(cfg: Config) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )


def fine_tune_model(
    model: Any,
    tokenizer: AutoTokenizer,
    train_examples: Sequence[Example],
    labels: Sequence[str],
    cfg: Config,
) -> Any:
    tokenizer.padding_side = "right"
    train_dataset = build_train_dataset(tokenizer, train_examples, labels, cfg.max_seq_len)

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    sft_config = SFTConfig(
        output_dir=os.path.join(cfg.output_dir, "adapter"),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        completion_only_loss=True,
        max_length=cfg.max_seq_len,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg),
    )
    trainer.train()
    trainer.save_model()

    trained_model = trainer.model
    if hasattr(trained_model, "gradient_checkpointing_disable"):
        trained_model.gradient_checkpointing_disable()
    trained_model.eval()
    tokenizer.padding_side = "left"
    trained_model.config.use_cache = True
    return trained_model


def write_results_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "label",
        "tp",
        "fp",
        "fn",
        "gold_count",
        "pred_count",
        "precision",
        "recall",
        "f1",
        "n_test_examples",
        "model_name",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = Config()
    setup_logging()
    set_seed(cfg.seed)
    random.seed(cfg.seed)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading datasets")
    train_examples = load_examples(
        cfg.train_path,
        limit=cfg.limit_train,
        strip_entity_whitespace=cfg.strip_entity_whitespace,
    )
    test_examples = load_examples(
        cfg.test_path,
        limit=cfg.limit_test,
        strip_entity_whitespace=cfg.strip_entity_whitespace,
    )
    labels = get_label_inventory()
    validate_examples_labels(train_examples, "train")
    validate_examples_labels(test_examples, "test")

    LOGGER.info(
        "Train examples: %d | Test examples: %d | Labels: %s",
        len(train_examples),
        len(test_examples),
        labels,
    )
    LOGGER.info("Loading model/tokenizer: %s", cfg.model_name)
    model, tokenizer = load_model_and_tokenizer(cfg.model_name, use_4bit=cfg.use_4bit)

    if cfg.max_new_tokens is None:
        cfg.max_new_tokens = min(
            compute_dynamic_max_new_tokens(
                tokenizer=tokenizer,
                train_examples=train_examples,
                cap=cfg.model_max_output_tokens,
            ),
            cfg.model_max_output_tokens,
        )
    else:
        cfg.max_new_tokens = min(cfg.max_new_tokens, cfg.model_max_output_tokens)
        LOGGER.info(
            "Using user-specified max_new_tokens=%d (capped at %d)",
            cfg.max_new_tokens,
            cfg.model_max_output_tokens,
        )

    base_pred_path = os.path.join(cfg.output_dir, "base_predictions.jsonl") if cfg.save_predictions else None
    ft_pred_path = os.path.join(cfg.output_dir, "finetuned_predictions.jsonl") if cfg.save_predictions else None

    LOGGER.info("Running zero-shot evaluation on test set")
    all_rows = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_examples=test_examples,
        labels=labels,
        max_new_tokens=cfg.max_new_tokens,
        repetition_penalty=cfg.repetition_penalty,
        stage="base",
        model_name=cfg.model_name,
        debug_alignment=cfg.debug_alignment,
        save_predictions_path=base_pred_path,
    )

    LOGGER.info("Fine-tuning with QLoRA/SFT")
    model = fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        labels=labels,
        cfg=cfg,
    )

    LOGGER.info("Running post-fine-tuning evaluation on test set")
    all_rows.extend(
        evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_examples=test_examples,
            labels=labels,
            max_new_tokens=cfg.max_new_tokens,
            repetition_penalty=cfg.repetition_penalty,
            stage="finetuned",
            model_name=cfg.model_name,
            debug_alignment=cfg.debug_alignment,
            save_predictions_path=ft_pred_path,
        )
    )

    write_results_csv(all_rows, cfg.results_csv)
    LOGGER.info("Saved results CSV to %s", cfg.results_csv)


if __name__ == "__main__":
    main()