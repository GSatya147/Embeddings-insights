import os, json

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

load_dotenv()
key=os.getenv("PINECONE_API_KEY")

UPSERT = False

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

if UPSERT:
    index.upsert(vectors=data_dict_field, namespace="honey-badger-wiki")

query = input(">> ")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query_embedding = model.encode(query)

query_results = index.query(
    namespace="honey-badger-wiki",
    vector= query_embedding, 
    top_k=3,
    include_metadata=True,
    include_values=True
)

for match in query_results.matches:
    print(f"Score: {match.score:.2f}")
    print(f"Content: {match.metadata["content"]}")
    print(f"{'-' * 50}")