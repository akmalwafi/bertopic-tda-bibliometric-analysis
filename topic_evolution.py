# topic_evolution.py
import os
import pandas as pd
from bertopic import BERTopic

OUT_DIR = "data/processed/bertopic"
MODEL_DIR = os.path.join(OUT_DIR, "bertopic_model")
DOC_TOPICS_CSV = os.path.join(OUT_DIR, "doc_topics.csv")

def main(top_n_topics=10):
    topic_model = BERTopic.load(MODEL_DIR)
    df = pd.read_csv(DOC_TOPICS_CSV)

    # ✅ REQUIRED columns
    required = ["cleaned", "topic", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in doc_topics.csv: {missing}")

    # Clean types
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["topic", "year"]).copy()

    df["topic"] = df["topic"].astype(int)
    df["year"] = df["year"].astype(int)

    # Optional: remove noise/outliers
    df = df[df["topic"] != -1].copy()

    docs = df["cleaned"].fillna("").astype(str).tolist()
    topics = df["topic"].tolist()
    timestamps = df["year"].tolist()

    if not docs:
        raise ValueError("No documents left after filtering. Check your CSV/topic filtering.")

    print("✅ Docs used:", len(docs))
    print("✅ Year range:", min(timestamps), "to", max(timestamps))

    # ✅ FIXED: correct argument order + pass topics by keyword
    tot = topic_model.topics_over_time(
        docs,
        timestamps,
        topics=topics,
        global_tuning=False,
        evolution_tuning=False
    )

    # Debug sanity checks
    print("✅ topics_over_time rows:", tot.shape[0])
    print(tot.head(10))

    os.makedirs(OUT_DIR, exist_ok=True)
    tot.to_csv(os.path.join(OUT_DIR, "topics_over_time.csv"), index=False)

    fig = topic_model.visualize_topics_over_time(tot, top_n_topics=top_n_topics)
    fig.write_html(os.path.join(OUT_DIR, "topics_over_time.html"))

    print("✅ Saved topics_over_time.html")

if __name__ == "__main__":
    main(top_n_topics=10)