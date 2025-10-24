import faiss
import numpy as np
import pandas as pd

def generate_indices(train_vectors, test_vectors, output_path, n_examples=5, d=768):
    pd.set_option('display.max_columns', None)

    quantizer = faiss.IndexFlatL2(d)
    nlist = len(test_vectors) // 39
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    index.train(train_vectors)
    index.add(train_vectors)

    # 查询：返回距离和对应向量的索引
    distances, indices = index.search(test_vectors, n_examples)
    np.save(output_path + "indices", indices)

    return indices