import os

import chroma_db
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

load_dotenv()
key=os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=key)

