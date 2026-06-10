import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def make_embeddings(
    input_csv="data/processed/cleaned_docs.csv",
    text_col="cleaned",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    out_dir="data/processed/embeddings",
    batch_size=64,
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str).tolist()

    empty_idx = [i for i, t in enumerate(texts) if not t.strip()]
    if empty_idx:
        print(f"⚠ Found {len(empty_idx)} empty documents. They will still be embedded (as empty strings).")

    print(f"📦 Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"🧠 Encoding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    emb_path = os.path.join(out_dir, "sbert_embeddings.npy")
    np.save(emb_path, embeddings)

    meta_path = os.path.join(out_dir, "meta.csv")
    df_out = df.copy()
    df_out["is_empty"] = [i in empty_idx for i in range(len(texts))]
    df_out.to_csv(meta_path, index=False)

    print("\n✅ Done!")
    print("Embeddings shape:", embeddings.shape)
    print("Saved:", emb_path)
    print("Saved:", meta_path)


if __name__ == "__main__":
    make_embeddings()