#### Migration from chroma Db to pinecone
1. Pull off the chroma Db using `get(include = ["documents", "embeddings", "metadatas"])`
2. Reformat into pinecone compatible, (id, values, metadata={with original content})
3. Create the pinecone index, (correct dimension + cosine similarity for semantic search)
4. Upsert in batches, as upsert accepts lists but capped at 100 docs.
5. Run queries 
6. Compare with chroma