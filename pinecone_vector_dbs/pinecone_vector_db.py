import os, json

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

load_dotenv()
key=os.getenv("PINECONE_API_KEY")

with open("./data.json", "r") as f:
    data_json = json.load(f)

data_dict_field: list[dict] = []

for id, embedding, document in zip(data_json["ids"], data_json["embeddings"], data_json["documents"]):
    field = {
        "id": id,
        "values": embedding,
        "metadata":{
            "source": "manual",
            "content": document
        }
    }
    data_dict_field.append(field)

pc = Pinecone(api_key=key)
index_name = "honey-badger"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
index.upsert(vectors=data_dict_field, namespace="honey-badger-wiki")

