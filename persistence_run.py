import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from umap import UMAP


def run_persistence(
    embeddings_path="data/processed/embeddings/sbert_embeddings.npy",
    out_dir="data/processed/tda",
    use_umap=True,
    umap_dim=2,
    max_points=1500,
    maxdim=2
):
    os.makedirs(out_dir, exist_ok=True)

    X = np.load(embeddings_path)

    # ✅ persistence is heavy → subsample for safety
    if X.shape[0] > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], max_points, replace=False)
        X = X[idx]

    # ✅ reduce dimension to make ripser fast
    if use_umap:
        X = UMAP(n_neighbors=15, n_components=umap_dim, metric="cosine", random_state=42).fit_transform(X)

    # ripser needs euclidean distances; UMAP output is euclidean-friendly
    result = ripser(X, maxdim=maxdim)
    dgms = result["dgms"]

    # ✅ Save diagram image
    plt.figure(figsize=(7, 6))
    plot_diagrams(dgms, show=False)
    plt.title("Persistence Diagram (Ripser)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "persistence_diagram.png"), dpi=220)
    plt.close()

    # ✅ Save barcode image
    plt.figure(figsize=(10, 5))
    plot_diagrams(dgms, show=False, lifetime=True)
    plt.title("Persistence Barcode / Lifetimes (Ripser)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "persistence_barcode.png"), dpi=220)
    plt.close()

    # ✅ Save raw arrays too (optional)
    np.save(os.path.join(out_dir, "persistence_dgms.npy"), np.array(dgms, dtype=object), allow_pickle=True)

    print("✅ Saved:")
    print(" -", os.path.join(out_dir, "persistence_diagram.png"))
    print(" -", os.path.join(out_dir, "persistence_barcode.png"))
    print(" -", os.path.join(out_dir, "persistence_dgms.npy"))


if __name__ == "__main__":
    run_persistence()