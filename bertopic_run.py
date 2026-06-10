import os
import numpy as np
import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from umap import UMAP
from hdbscan import HDBSCAN


def run_bertopic(
    cleaned_csv="data/processed/cleaned_docs.csv",
    embeddings_path="data/processed/embeddings/sbert_embeddings.npy",
    text_col="cleaned",
    out_dir="data/processed/bertopic",
):
    os.makedirs(out_dir, exist_ok=True)

    # Load cleaned documents
    df = pd.read_csv(cleaned_csv)
    docs = df[text_col].fillna("").astype(str).tolist()

    # Load SBERT embeddings
    embeddings = np.load(embeddings_path)
    if len(docs) != embeddings.shape[0]:
        raise ValueError(f"Docs count ({len(docs)}) != embeddings rows ({embeddings.shape[0]})")

    print(f"📄 Loaded {len(docs)} documents")
    print(f"🧠 Embeddings shape: {embeddings.shape}")

    # Vectorizer for topic keywords
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    # Tune UMAP
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    # Tune HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    topic_model = BERTopic(
        language="english",
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Save document-topic assignments
    df_out = df.copy()
    df_out["topic"] = topics
    df_out.to_csv(os.path.join(out_dir, "doc_topics.csv"), index=False)

    # Topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(out_dir, "topic_info.csv"), index=False)

    # Save keywords per topic
    with open(os.path.join(out_dir, "topics_keywords.txt"), "w", encoding="utf-8") as f:
        for topic_id in topic_info["Topic"]:
            if topic_id == -1:
                continue
            words = topic_model.get_topic(topic_id)
            f.write(f"Topic {topic_id}:\n")
            f.write(", ".join([w for w, _ in words[:15]]) + "\n\n")

    # Save model
    topic_model.save(os.path.join(out_dir, "bertopic_model"))

    # Summary
    n_topics_excl_noise = (topic_info["Topic"] != -1).sum()
    n_noise = (df_out["topic"] == -1).sum()
    print("\n✅ BERTopic finished!")
    print("Topics (excluding -1):", n_topics_excl_noise)
    print("Noise docs (-1):", n_noise)

    # ✅ Visualizations
    # ✅ Visualizations
    try:
        # Show ALL topics (exclude topic -1)
        # Show ALL topics (exclude topic -1)
        fig_all = topic_model.visualize_barchart(top_n_topics=n_topics_excl_noise)

        # ✅ Good behavior:
        # - few topics -> NOT giant
        # - many topics -> scrollable
        base_h = 900
        per_topic_h = 90  # smaller per-topic height so it won't blow up
        max_h = 12000  # cap so browser doesn't go crazy
        height = min(max_h, base_h + per_topic_h * n_topics_excl_noise)

        fig_all.update_layout(
            autosize=True,
            width=None,
            height=height,
            margin=dict(l=10, r=10, t=70, b=10),
            title=dict(text="Topic Word Scores", x=0.5, xanchor="center"),
            font=dict(size=11),
            bargap=0.25
        )

        fig_all.write_html(
            os.path.join(out_dir, "barchart_all_topics.html"),
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True}
        )

        # Topic map (works best when >=3 topics)
        if n_topics_excl_noise >= 3:
            fig2 = topic_model.visualize_topics()

            # Make responsive + bigger (this is the important one)
            fig2.update_layout(
                autosize=True,
                width=None,
                height=950,  # try 900–1100
                margin=dict(l=10, r=10, t=80, b=10),
                title=dict(text="Intertopic Distance Map", x=0.5, xanchor="center")
            )

            fig2.write_html(
                os.path.join(out_dir, "topics_map.html"),
                include_plotlyjs="cdn",
                full_html=True,
                config={"responsive": True}
            )
        else:
            print("⚠ Skipping topics_map (need >= 3 topics)")

        print("📊 Visualizations saved (ALL topics)")
    except Exception as e:
        print("⚠ Visualization skipped:", e)

    return topic_model


if __name__ == "__main__":
    run_bertopic()