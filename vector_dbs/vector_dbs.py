from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

embedding_client = genai.Client()

result = embedding_client.models.embed_content(
                model="gemini-embedding-001",
                contents=["Hi, i'm satya"],
                config=types.EmbedContentConfig(output_dimensionality=3072)
            )

print(result.embeddings[0].values)

