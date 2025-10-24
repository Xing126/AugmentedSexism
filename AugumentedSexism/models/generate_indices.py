import faiss
import numpy as np
import pandas as pd

def generate_indices(train_path, test_path, output_file, n_examples=5, d=768):
    pd.set_option('display.max_columns', None)
    train_vectors = np.load(train_path)
    test_vectors = np.load(test_path)

    quantizer = faiss.IndexFlatL2(d)
    nlist = max(1, len(test_vectors) // 39)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    index.train(train_vectors)
    index.add(train_vectors)

    distances, indices = index.search(test_vectors, n_examples)
    np.save(output_file, indices)

if __name__ == "__main__":
    print(np.load("../data/processed/indices.npy")[0])
    print(np.array([1,2,3,4,5]))
