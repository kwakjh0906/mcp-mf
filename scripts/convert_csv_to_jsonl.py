from pathlib import Path
import csv
import json

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "manufacturing_terms.csv"
JSONL_PATH = BASE_DIR / "data" / "manufacturing_terms.jsonl"


def csv_to_jsonl():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    rows = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                f"용어: {row['term']}\n"
                f"카테고리: {row['category']}\n"
                f"정의: {row['definition']}\n"
                f"연관어: {row['related']}\n"
                f"키워드: {row['keywords']}\n"
            )
            rows.append(
                {
                    "term": row["term"],
                    "category": row["category"],
                    "definition": row["definition"],
                    "related": row["related"],
                    "keywords": row["keywords"],
                    "text": text,
                }
            )

    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"CSV → JSONL 완료: {JSONL_PATH}")


if __name__ == "__main__":
    csv_to_jsonl()
