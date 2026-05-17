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
**Persistent:** ChromaDB writes the index and the raw data to disk specifically to a folder you point it at. You embed 50 documents, store them, close the terminal. Next run: ChromaDB reads the index back from disk. Your data is still there. You can query it immediately without re-embedding.  
- Embedding costs money (API calls) and time, If you're using OpenAI embeddings, you don't want to re-embed your corpus every time you run the script. Persistent mode means you embed once, store, query forever.  
- This is also why production RAG systems have an ingestion pipeline (embed + store, runs once or on schedule) completely separate from the query pipeline (load from DB, search, return, runs on every user request).  

3. How data is stored in a vector Db?  
- Vector Dbs always store vector embeddgings alongside the metadata (time or year, word count, index position, Source) instead of pure vectors.  
**Why?**  
- *Retrieval of the original content:* The vector is a pointer, not the thing. When ChromaDB returns a result, you need the original text to show the user.  That text lives in the metadata (or ChromaDB's document store, same idea).  
- *Structured filtering before or after vector search:* This is huge. Imagine a legal document search tool. User asks "find clauses about liability." You could return the 5 most semantically similar chunks but what if the user only wants clauses from contracts signed after 2023? Vector similarity has no concept of dates. Metadata filtering does, You say: search by similarity AND filter where year >= 2023. This is called filtered ANN search and it's a core feature of every production vector DB.  
- *Debugging and evaluation:* When retrieval goes wrong, metadata (chunk index, source file, word count) is how you diagnose which chunk was returned and why it might be a bad match. You can't debug a raw vector.  

#### Is it worth the hassel?
- Earlier models used to have less context window (4k,10k..), but thanks to today's computing architecture we have 1Mn+ tokens context in some models.  
- Hence choosing the right stack is very important cus vector storing and RAGs introduce a whole layer of architecture on top our core functionality layer.  
- As the architectural implementations are expensive decisions are made based on circumstances  
1. **Enterprise:** since enterprise's data lakes are mostly huge, these documents in the context window can cause serious issues like: hallucination due to more noise, API costs due to documents eating the tokens in every turns. RAGs architecture is massively useful and vector DBs are the only viable choice here.  
2. **Start-ups, ventures with feasible data:** usually in this type of scenarios the actual API costs and hallucination cost lesser than the actual impleemntation layer of RAGs. Maintaing *model simplicity* is better approach than vector Dbs, RAGs stack..
- RAG: less noise to the model (the niddle in the middle of haystack)
- But it has it's own limitations like 95% accuracy due to vector DBs instead of depending on the model's attention mechanism, which further decreases the accuracy once the model responses.  

#### RAG
```bash
RAG: documents (large) → chunking → embedding model → vector embeddings → vector DBs (metadata + embeddings) → re-ranking → user prompt → approx similar data retrieval`
```

- remaining insights complete IBM introductory video