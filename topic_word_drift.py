# topic_change_report.py
import os
import pandas as pd
from bertopic import BERTopic

OUT_DIR = "data/processed/bertopic"
MODEL_DIR = os.path.join(OUT_DIR, "bertopic_model")
DOC_TOPICS_CSV = os.path.join(OUT_DIR, "doc_topics.csv")


def jaccard_distance(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / len(a | b))


def top_words_for_group(agg_text, vectorizer, ctfidf_model, top_n=8):
    X = vectorizer.transform([agg_text])
    ctfidf = ctfidf_model.transform(X)
    scores = ctfidf.toarray().flatten()
    terms = vectorizer.get_feature_names_out()

    top_idx = scores.argsort()[::-1][:top_n]
    return [terms[i] for i in top_idx if scores[i] > 0]


def main(top_n_words=8, min_docs_per_year=3, max_rows_per_topic=999):
    topic_model = BERTopic.load(MODEL_DIR)
    df = pd.read_csv(DOC_TOPICS_CSV)

    for col in ["cleaned", "topic", "year"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in doc_topics.csv")

    df["topic"] = pd.to_numeric(df["topic"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["topic", "year"]).copy()
    df["topic"] = df["topic"].astype(int)
    df["year"] = df["year"].astype(int)

    # Remove outliers/noise
    df = df[df["topic"] != -1].copy()

    vectorizer = topic_model.vectorizer_model
    ctfidf_model = topic_model.ctfidf_model

    # Compute "topic name" (top words) per (topic, year)
    topic_year_words = {}
    topic_year_counts = {}

    for (t, y), sub in df.groupby(["topic", "year"]):
        if len(sub) < min_docs_per_year:
            continue
        agg_text = " ".join(sub["cleaned"].fillna("").astype(str).tolist())
        words = top_words_for_group(agg_text, vectorizer, ctfidf_model, top_n=top_n_words)
        topic_year_words[(t, y)] = words
        topic_year_counts[(t, y)] = len(sub)

    rows = []
    for t in sorted(df["topic"].unique()):
        years = sorted([y for (tt, y) in topic_year_words.keys() if tt == t])
        if len(years) < 2:
            continue

        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            w1 = topic_year_words[(t, y1)]
            w2 = topic_year_words[(t, y2)]

            added = [w for w in w2 if w not in w1]
            removed = [w for w in w1 if w not in w2]
            drift = jaccard_distance(w1, w2)

            rows.append({
                "Topic ID": t,
                "From Year": y1,
                "To Year": y2,
                "Docs (From)": topic_year_counts[(t, y1)],
                "Docs (To)": topic_year_counts[(t, y2)],
                "Topic Name (From)": " ".join(w1),
                "Topic Name (To)": " ".join(w2),
                "Added Words": ", ".join(added) if added else "-",
                "Removed Words": ", ".join(removed) if removed else "-",
                "Drift Score (0 stable → 1 changed)": round(drift, 3),
            })

        # optional: limit how many transitions shown per topic
        if max_rows_per_topic != 999:
            rows = rows[:max_rows_per_topic]

    out_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, "topic_change_report.csv")
    html_path = os.path.join(OUT_DIR, "topic_change_report.html")

    out_df.sort_values(["Drift Score (0 stable → 1 changed)"], ascending=False).to_csv(csv_path, index=False)

    # Make a readable HTML table
    html = out_df.sort_values(
        ["Topic ID", "From Year"]
    ).to_html(index=False, escape=False)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<h2>Topic Change Report (Word Drift)</h2>")
        f.write("<p>Shows how each topic's top words (topic name) changes between years.</p>")
        f.write(html)

    print("✅ Saved:", csv_path)
    print("✅ Saved:", html_path)


if __name__ == "__main__":
    main(top_n_words=8, min_docs_per_year=3)