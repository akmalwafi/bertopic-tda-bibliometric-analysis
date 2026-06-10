# preprocess_bib.py
import os
import pandas as pd
from bib_reader import load_multiple_bibs


def preprocess_bib(input_bib_paths, output_csv="data/processed/cleaned_docs.csv"):
    entries = load_multiple_bibs(input_bib_paths)

    rows = []

    for entry in entries:
        title = entry.get("title", "").strip()
        abstract = entry.get("abstract", "").strip()
        keywords = entry.get("keywords", "").strip()
        year = entry.get("year", "").strip()

        raw_text = " ".join([title, abstract, keywords]).strip()

        rows.append({
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "year": year,
            "cleaned": raw_text
        })

    df = pd.DataFrame(rows)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ Saved merged cleaned docs to {output_csv}")
    print(f"✅ Total entries: {len(df)}")