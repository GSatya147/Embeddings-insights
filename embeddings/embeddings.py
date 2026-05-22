from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

load_dotenv()


# Similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    mag_prod = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    return dot_product / mag_prod


# Create dataframe
def create_dataframe(embeddings):
    size = len(embeddings)
    df = np.zeros((size, size))

    for i in range(0, size):
        for j in range(0, size):
            df[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    pdf = pd.DataFrame(
        df,
        index=np.array([i + 1 for i in range(0, size)]),
        columns=np.array([i + 1 for i in range(0, size)]),
    )
    return pdf


# local transformer
def local_embed_model(sentences):
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        result = model.encode(sentences)
        return result

    except Exception as e:
        print(e)


# gemini embedding model/client
def gemini_embed_model(sentences):
    try:
        client = genai.Client()

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=sentences,
            config=types.EmbedContentConfig(output_dimensionality=3072),
        )

        return [embedding.values for embedding in result.embeddings]

    except Exception as e:
        print(e)


def compare_embeddings(cos_local, cos_gemini):

    assert cos_local.shape == cos_gemini.shape, "DataFrames must have the same shape"

    ids = np.triu_indices(cos_local.shape[0], k=1)
    rows, cols = ids[0] + 1, ids[1] + 1

    df = pd.DataFrame(
        {
            "sentence_i": rows,
            "sentence_j": cols,
            "cos_local": cos_local.values[ids],
            "cos_gemini": cos_gemini.values[ids],
        }
    )

    df["divergence"] = (
        df["cos_local"] - df["cos_gemini"]
    ).abs()  # How much they disagree
    df["bias"] = (
        df["cos_local"] - df["cos_gemini"]
    )  # +ve = local more generous gives who's systematically off
    df["consensus_floor"] = df[["cos_local", "cos_gemini"]].min(
        axis=1
    )  # What we can trust without picking a winner

    return df.sort_values("divergence", ascending=False).reset_index(drop=True)


if __name__ == "__main__":

    # Create sentences for embeddings
    sentences = [
        "How do I restart a server?",  # 1
        "The system is highly available",  # 2
        "retrieval augmented generation.",  # 3
        "How to bake sourdough bread",  # 4
        "What is the capital of France?",  # 5
        "Dogs chase cats",  # 6
        "The model does NOT perform well on long documents",  # 7
        "A man is playing guitar",  # 8
        "All birds can fly",  # 9
        "memory leak",  # 10
        "The HNSW index rebuild caused latency spikes",  # 11
        "machine learning",  # 12
        "RAG hallucination causes",  # 13
        "why is my chatbot making stuff up",  # 14
        "how to handle KeyError in Python dict",  # 15
        "latest transformer architecture",  # 16
        "Retriever, Chunker, Reranker, Generator",  # 17
        "fast retrieval at the cost of accuracy",  # 18
        "embedding model",  # 19
        "what happens to retrieval quality when chunk size increases beyond context window?",  # 20
    ]

    try:
        embeddings_local = local_embed_model(sentences=sentences)
        embeddings_gemini = gemini_embed_model(sentences=sentences)

    except Exception as e:
        print(e)

    for i, j in enumerate(sentences):
        print(f"{i+1}. {j}")

    data_local = create_dataframe(embeddings=embeddings_local)
    print(f"\nCosine Similarity table - local model: \n{data_local}")

    data_gemini = create_dataframe(embeddings=embeddings_gemini)
    print(f"\nCosine Similarity table - gemini model: \n{data_gemini}")

    insights = compare_embeddings(data_local, data_gemini)
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        print(insights)