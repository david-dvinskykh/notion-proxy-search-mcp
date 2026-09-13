#!/usr/bin/env python3
"""Regenerate the tokenizer golden file from a Hugging Face checkpoint.

The Go tokenizer reimplements XLM-RoBERTa's Unigram SentencePiece without the
precompiled charsmap, so its agreement with the reference has to be measured
rather than assumed. This script writes the reference segmentation for every
case in the corpus; internal/embed/tokenizer_golden_test.go compares against it.

    pip install tokenizers
    python3 scripts/tokenizer_reference.py <model dir> [corpus.txt ...]

The model directory is the checkpoint (it needs tokenizer.json). Extra corpus
files are read one case per line and appended to the built-in edge cases.
"""

import json
import pathlib
import sys

from tokenizers import Tokenizer

NBSP = chr(0x00A0)
ZWSP = chr(0x200B)
ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)
BOM = chr(0xFEFF)
ESC = chr(0x001B)
COMBINING_BREVE = chr(0x0306)

# Edge cases worth keeping forever: each one caught a real divergence or guards
# a rule that was read off the reference instead of guessed.
EDGE_CASES = [
    "",
    " ",
    "\n\t",
    "   пробелы   по   краям   ",
    "текст ",
    " текст",
    NBSP + "неразрывный" + NBSP + "пробел" + NBSP,
    "zero" + ZWSP + "width",
    "zero" + ZWNJ + "width" + ZWJ + "joiner",
    BOM + "bom",
    "a" + ESC + "b",
    "Привет мир",
    "query: сколько памяти ест контейнер gateway",
    "passage: Контейнер gateway на Raspberry Pi 4B держит 3,1–3,3 ГБ из 8 ГБ ОЗУ",
    "§16. Обновление правил",
    "§0 §2 §13.2 §16.4",
    "Полив теплиці працює за датчиком вологості, а не за розкладом",
    "Zażółć gęślą jaźń",
    "Zamówienie zostało wysłane do Krakowa kurierem",
    "Заказ ORD-2026-0041, счёт INV-7781",
    "The quick brown fox jumps over the lazy dog",
    "Файл: 2026-09-12 · Документ · Договор · приложение.pdf",
    "notion-query-data-sources и notion-fetch",
    "🧩 Заметки — 📘 Регламент проекта v3.1",
    "ООО «Ромашка», д.5, кв.12",
    "ё Ё й Й ї Ї є Є ґ Ґ",
    "й vs и" + COMBINING_BREVE,
    "ＦＵＬＬＷＩＤＴＨ",
    "½ ² ﬁ",
    "смешанныйTextСлитно123",
    "C++ и C# в Java/Spring",
    '"кавычки" и (скобки) [и] {ещё}',
    "SELECT url,\"Правило\" FROM \"collection://7c1e4a90\" WHERE \"Статус\" = 'действует'",
    "date:Замечено:start >= '2026-09-01' AND \"Ядро\" = '__YES__'",
    "Доверие: подтверждено · Тип: атрибут · Слой: активное",
]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    model_dir = pathlib.Path(argv[1])
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    cases: list[str] = []
    seen: set[str] = set()
    for case in EDGE_CASES:
        if case not in seen:
            seen.add(case)
            cases.append(case)
    for path in argv[2:]:
        for line in pathlib.Path(path).read_text().split("\n"):
            line = line.strip()
            if line and len(line) <= 400 and line not in seen:
                seen.add(line)
                cases.append(line)

    out = [{"text": c, "ids": tok.encode(c).ids} for c in cases]
    target = pathlib.Path(__file__).resolve().parent.parent / "internal/embed/testdata/tokenizer_golden.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {target} with {len(out)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
