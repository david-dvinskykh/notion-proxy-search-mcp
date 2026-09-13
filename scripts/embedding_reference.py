#!/usr/bin/env python3
"""Regenerate the embedding golden file from the checkpoint's own ONNX export.

The Go encoder runs int8 weights with dynamically quantized activations. Its
agreement with the float32 model has to be measured against an implementation
that shares nothing with it, so the reference here is onnxruntime running the
export that ships with the checkpoint.

    pip install onnxruntime numpy tokenizers
    python3 scripts/embedding_reference.py <model dir>

<model dir> needs tokenizer.json and onnx/model.onnx (or model.onnx beside it).
internal/embed/embedding_golden_test.go compares the Go vectors against the
result.
"""

import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Cases span the languages and shapes the workspace actually holds, plus the
# near-duplicate pairs that make a ranking error visible.
CASES = [
    ("query", "сколько памяти ест контейнер gateway"),
    ("query", "потребление ОЗУ на домашнем сервере"),
    ("query", "по какому признаку включается полив"),
    ("query", "як налаштувати полив у теплиці"),
    ("query", "how much RAM does the container use"),
    ("passage", "Контейнер gateway на Raspberry Pi 4B держит 3,1 ГБ из 8 ГБ ОЗУ"),
    ("passage", "Полив теплицы идёт по датчику влажности, а не по расписанию"),
    ("passage", "Расходы по проекту разносятся по категориям в конце месяца"),
    ("passage", "Сервер наблюдения работает на Raspberry Pi (aarch64, 4 CPU, 8 ГБ)"),
    ("passage", "Полив теплиці працює за датчиком вологості, а не за розкладом"),
    ("passage", "Zamówienie zostało wysłane do Krakowa kurierem"),
    ("passage", "The mirror exposes every Notion data source as a SQLite view"),
    ("passage", "§16. Обновление правил: повторное замечание меняет правило"),
    ("passage", "SELECT url,\"Правило\" FROM \"collection://7c1e4a90\" WHERE \"Статус\" = 'действует'"),
    ("passage", "кот"),
    ("passage", "кошка"),
    ("passage", "бетон"),
    ("passage", "2026-09-12"),
    ("passage", "🧩 Заметки — 📘 Регламент проекта v3.1"),
    ("passage", "x"),
    ("passage", " ".join(["длинный"] * 120)),
]

PREFIX = {"query": "query: ", "passage": "passage: "}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    model_dir = pathlib.Path(argv[1])
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    onnx_path = model_dir / "onnx" / "model.onnx"
    if not onnx_path.exists():
        onnx_path = model_dir / "model.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {i.name for i in session.get_inputs()}

    # The position table caps the sequence; the Go side truncates the same way.
    max_tokens = 512

    out = []
    for mode, text in CASES:
        ids = tok.encode(PREFIX[mode] + text).ids[:max_tokens]
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros((1, len(ids)), dtype=np.int64)
        hidden = session.run(None, feed)[0][0]          # [tokens, hidden]
        pooled = hidden.mean(axis=0)                     # mean pooling, mask is all ones
        pooled /= np.linalg.norm(pooled)
        out.append({
            "mode": mode,
            "text": text,
            "tokens": len(ids),
            "vector": [round(float(v), 5) for v in pooled],
        })

    target = pathlib.Path(__file__).resolve().parent.parent / "internal/embed/testdata/embedding_golden.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {target} with {len(out)} cases, {len(out[0]['vector'])} dims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
