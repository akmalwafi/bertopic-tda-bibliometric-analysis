from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

emb = np.load("data/processed/embeddings/sbert_embeddings.npy")

sim_matrix = cosine_similarity(emb)

doc_id = 0
similarities = sim_matrix[doc_id]

top_idx = similarities.argsort()[::-1][:5]

print("Top similar docs to doc 0:")
for idx in top_idx:
    print(idx, similarities[idx])
