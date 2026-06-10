import os
import pandas as pd
from bertopic import BERTopic


def apply_topic_labels(
    model_dir="data/processed/bertopic/bertopic_model",
    labels_csv="data/processed/bertopic/topic_labels.csv",
    out_dir="data/processed/bertopic"
):
    os.makedirs(out_dir, exist_ok=True)

    print("📦 Loading BERTopic model...")
    topic_model = BERTopic.load(model_dir)

    print("🏷 Loading topic labels...")
    labels_df = pd.read_csv(labels_csv)

    # make sure topic ids are int
    labels_df["Topic"] = labels_df["Topic"].astype(int)

    # build mapping: topic id -> human-readable label
    label_map = dict(zip(labels_df["Topic"], labels_df["Topic_Label"]))

    print("✅ Applying labels...")
    topic_model.set_topic_labels(label_map)

    topic_info = topic_model.get_topic_info()
    n_topics_excl_noise = (topic_info["Topic"] != -1).sum()

    print("📊 Regenerating labeled barchart...")
    fig_all = topic_model.visualize_barchart(
        top_n_topics=n_topics_excl_noise,
        custom_labels=True
    )

    base_h = 900
    per_topic_h = 90
    max_h = 12000
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

    if n_topics_excl_noise >= 3:
        print("🗺 Regenerating labeled topic map...")
        fig2 = topic_model.visualize_topics(custom_labels=True)

        fig2.update_layout(
            autosize=True,
            width=None,
            height=950,
            margin=dict(l=10, r=10, t=80, b=10),
            title=dict(text="Intertopic Distance Map", x=0.5, xanchor="center")
        )

        fig2.write_html(
            os.path.join(out_dir, "topics_map.html"),
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True}
        )

    print("✅ Done. Labeled HTML files saved in:", out_dir)


if __name__ == "__main__":
    apply_topic_labels()