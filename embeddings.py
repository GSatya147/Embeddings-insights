import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Create sentences for embeddings
sentence: list[str] = [
    "I'm a good person",
    "everybody calls me nice",
    "Hope you had a wonnderful weekend!",
    "i'm a good person",
]

# Similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    mag_prod = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    return dot_product/mag_prod   

# Create dataframe
def create_dataframe(embeddings):
    size = size(embeddings)
    df = np.zeros((size, size))

    for i in range(0, size):
        for j in range(0, size):
            df[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    pdf = pd.DataFrame(df)
    return pdf

# try/except block for transformer handling
try:
    # Select the model from package
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    embeddings = model.encode(sentence)
    print(embeddings)

    data = create_dataframe(embeddings=embeddings)
    print(data)

except Exception as e:
    print(e)
