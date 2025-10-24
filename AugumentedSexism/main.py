import os

from openai import api_key

from models.generate_vectors import dataset_vectorize
from models.generate_indices import generate_indices
from models.ai_request import *


# ================= CONFIGURATION =================
api_url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
api_key="fb9f6828-c8d7-4f87-a038-6145b078a48b",
model="doubao-1-5-lite-32k-250115"

train_data = "./data/raw/small_data.csv"
test_data = "./data/raw/test_data.csv"

# ==================== DEFAULT ====================
train_vectors = "./data/processed/train_vectors.npy"
test_vectors = "./data/processed/test_vectors.npy"
indices = "./data/processed/indices.npy"
rules1 = ["sexism", "gender", "hostile"]
rules2 = ["sexism", "gender", "hostile", "misogyny", "misandry"]
augumented_data = "./data/processed/augumented_data.csv"

# 1. generate vectors
if not os.path.exists(train_vectors):
    dataset_vectorize(train_data, train_vectors)
if not os.path.exists(test_vectors):
    dataset_vectorize(test_data, test_vectors)



# 2. generate indices
if not os.path.exists(indices):
    n_examples = 5  # The amount of examples
    d = 768 # The same as first step's dimension
    generate_indices(train_vectors,
                     test_vectors,
                     indices,
                     n_examples=n_examples,
                     d=d)


# 3. ai_request
example_producer = ExampleProducer(indices, train_data)
text_producer = TextProducer(test_data)
api_client = APIClient(api_key, api_url, model)
text_handler = TextHandler(augumented_data)
model = AugumentedSexismModel(example_producer, text_producer, api_client, text_handler,rules=rules1)
try:
    model.main()
except KeyboardInterrupt:
    model.save_dataset()
