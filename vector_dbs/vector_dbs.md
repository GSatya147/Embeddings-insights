#### Pre-questions 
1. Since we learnt cosine similarity how is it managed at scale?

- To retrieve relaevant data from query we need similarity techniques to search.  
- Well even if we have cosine similarity we cannot do a linear search at scale say among 50 million vectors we cannot perform cos similarity at linear scale. It gets awfully slow.  
- Solution is **vector indexing:** *Approximate Nearest Neighbour (ANN)* search, where a tiny bit of accuracy is traded (95%) with latency (ms), it gives approximate similarities. In practice 95% accuracy is good enough for retrieval tasks.  
- One of the algorithms used is *Hierarchical Navigable Small World (HNSW)*.  
**Intuition:** imagine looking for someone in a city. Instead of knocking on every door, navigate a hierarchy i.e. country → city → neighbourhood → street → house. HNSW builds a graph of vectors at multiple "zoom levels." At search time enter at the top (coarse), navigate toward the query fast, then zoom in for precision.  
**result**: search that's logarithmic, not linear. Millions of vectors, still milliseconds.   
- ChromaDB uses HNSM under the hood.  
- Other approaches/alogrithms include *Inverted Far Index (IVF)*.

2. In-memory vs persistent storge?  

- ChromaDB stores vectors persistently.  
**In-memory:** ChromaDB creates the collection, builds the HNSW index, holds everything in RAM. You embed 50 documents, store them, query them — works great. You close the terminal. Everything is gone. Next run: start from zero.  
**Persistent:** ChromaDB writes the index and the raw data to disk — specifically to a folder you point it at. You embed 50 documents, store them, close the terminal. Next run: ChromaDB reads the index back from disk. Your data is still there. You can query it immediately without re-embedding.  
- embedding costs money (API calls) and time. If you're using OpenAI embeddings, you don't want to re-embed your corpus every time you run the script. Persistent mode means you embed once, store, query forever.
- This is also why production RAG systems have an ingestion pipeline (embed + store, runs once or on schedule) completely separate from the query pipeline (load from DB, search, return, runs on every user request).

3. How data is stored in a vector Db?
- Vector Dbs always store vector embeddgings alongside the metadata (time or year, word count, index position, Source) instead of pure vectors.
**Why?**
- Retrieval of the original content:  The vector is a pointer, not the thing. When ChromaDB returns a result, you need the original text to show the user.  That text lives in the metadata (or ChromaDB's document store, same idea).



