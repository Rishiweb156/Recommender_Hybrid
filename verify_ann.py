# verify_ann.py
import numpy as np
from src.ann import AnnIndex

def main():
    # small synthetic embeddings
    np.random.seed(0)
    emb = np.random.randn(1000, 128).astype('float32')
    ann = AnnIndex(emb, index_type='auto', metric='cosine')
    q = emb[5].reshape(1, -1)
    ids, dists = ann.query(q, top_k=5)
    print("Neighbor IDs:", ids)
    print("Distances/similarities:", dists)

if __name__ == "__main__":
    main()
