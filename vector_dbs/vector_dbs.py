from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb

load_dotenv()

embedding_client = genai.Client()

result = embedding_client.models.embed_content(
                model="gemini-embedding-001",
                contents=["Hi, i'm satya"],
                config=types.EmbedContentConfig(output_dimensionality=3072)
            )

print(result.embeddings[0].values)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection - where documents, metadata, indexes are stored (top level container like index in pinecone)
collections = chroma_client.get_or_create_collection(name="my_collection")

collections.add(
    # index=[]
)