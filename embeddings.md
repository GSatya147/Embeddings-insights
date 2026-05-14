#### Terminology
- **Embedding:** Process of mapping the word -> vector embeddings

- **Embedding model:** A specialised transformer which trained explicitly to turn tokens into vector embeddings

- **Vector embeddings/ Embeddings:** Vector/list of co-ordinates of a word or position of a word in the implicit/imaginary high dimensional space 

- **Embedding space:** A high level space which captures the semantic meaning and context where each direction reveals the context, similar words are tend to cluster together. (Mind the space is just the sake of understanding there is no explicit storage here and the embeddins are not samrt enough to create and place the words in the sense of direction, they are just numericals and and when we try to arrange the positions simialar words tend to cluster we later name the understanding direction convention.)

#### Mental Model:
```bash
user -> Embedding model -> (input) word/token -> (Embedding process) -> (output) Vector embeddings -> positioning the coordinates -> creates -> high level dimension called Embedding space
```

#### Similarity:
- Here the magnitude (length) doesn't actuall reveal anything to be precise is not much of a needy information.
- To determine how contextually similar two words are, we need angle between the two vectors (right?). Think about it, angle reveals how far or close the two vectors are pointing with respective each other. 

Hence, cosine_similarity is used, cosine based similarity i.e.
`cosine_similarity = dot(A.B)/(mag(A) x Mag(B))`, looks familiar? 

this reveals the radians or angle between the vectors, well..not clearly. Do you remember the cos table values, cos range is [-1,1] thats what cosine_similarity actually reveals. 

```bash
closer 0        -> perpendicular
closer to 1     -> Same direction
closer to -1    -> opposite direction
```

But realistically in practice embeddings models tends to produce positive vectors, cus two words in a sense would never be truly opposite there will be a sense of direction which will/can capture a similarity, say `hot` x `cold` but they can be placed in same direction of `temperature`, right?
