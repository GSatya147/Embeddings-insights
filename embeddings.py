import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Create sentences for embeddings
sentence: list[str] = [
    "I'm a good person",
    "everybody calls me nice",
    "Hope you had a wonnderful weekend!",
]

# Similarity
def cosine_similarity(embeddings):
    for i in range(0, embeddings):
        for j in range(1, embeddings):
            dot_product = np.dot(embeddings[i], embeddings[j])
            mag_prod = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])

            return dot_product/mag_prod

# def create_df(similarity_list):

try:
    # Select the model from package
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Calculating embeddings
    embeddings = model.encode(sentence)
    print(embeddings)

    # create pandas dataframe
    df = pd.DataFrame(
        
    )

except Exception as e:
    print(e)


print(cosine_similarity(embeddings=embeddings))
