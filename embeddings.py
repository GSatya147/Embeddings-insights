import numpy as np
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
    mag_prod = np.linalg.norm(vector_a) * np.linalg.norm(vector_a)

    return dot_product/mag_prod   

# Create table
def print_table(embeddings):
    for i in range(0,len(embeddings)):
        print("\t\t{}".format(f'{i+1}'), end="")
    print()

    for i in range(0, len(embeddings)):
        print(f"\t{i+1}", end="")
        for j in range(0, len(embeddings)):
            print(f"\t{cosine_similarity(embeddings[i], embeddings[j]):.3f}\t", end="")
        print()

try:
    # Select the model from package
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Calculating embeddings
    embeddings = model.encode(sentence)
    print(embeddings)

    # Print similarity table
    print_table(embeddings=embeddings)

except Exception as e:
    print(e)
