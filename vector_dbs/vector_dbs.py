from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb

import os

load_dotenv()

sentences = [
    "I'm a good guy",
    "Everyone calls me nice",
    "New delhi is India's capital",
]

client = genai.Client()

result = client.models.embed_content(
                model=os.getenv("MODEL"),
                contents=sentences,
                config=types.EmbedContentConfig(output_dimensionality=3072)
            )

# print(result.embeddings[0].values)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection - where documents, metadata, indexes are stored (top level container like index in pinecone)
collections = chroma_client.get_or_create_collection(name="my_collection")

collections.add(
    ids=[f"doc_{i+1}" for i in range(len(result.embeddings))],
    embeddings=[e.values for e in result.embeddings]
)

# Query
query = input(">> ")

# Query embedding
query_embedding = client.models.embed_content(
                    model=os.getenv("MODEL"),
                    contents=[query],
                    config=types.EmbedContentConfig(output_dimensionality=3072)
                )

list_1: list = [query_embedding.embeddings[0].values]
print(type(list_1))

try:
    query_result = collections.query(
                    query_embeddings=[list_1],
                    n_results=3,
                )
except Exception as e:
    print(e)

print(query_result)