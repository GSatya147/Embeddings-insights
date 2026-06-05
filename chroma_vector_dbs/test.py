# # chunking test
# Hone_badger = """
# The honey badger has a fairly long body but is distinctly thick-set and broad across the back. Its skin is remarkably loose, and allows the animal to turn and twist freely within it. The skin around the neck is 6 mm (0.24 in) thick, an adaptation to fighting conspecifics. The head is small and flat, with a short muzzle. The eyes are small, and the ears are little more than ridges on the skin, another possible adaptation to avoiding damage while fighting. The honey badger has short and sturdy legs, with five toes on each foot. The feet are armed with very strong claws, which are short on the hind legs and remarkably long on the forelimbs. It is a partially plantigrade animal whose soles are thickly padded and naked up to the wrists. The tail is short and is covered in long hairs, save for below the base.

# The honey badger is the largest terrestrial mustelid in Africa. Adults measure 23 to 28 cm (9.1 to 11.0 in) in shoulder height and 55–77 cm (22–30 in) in body length, with the tail adding another 12–30 cm (4.7–11.8 in). Females are smaller than males. In Africa, males weigh 9 to 16 kg (20 to 35 lb) while females weigh 5 to 10 kg (11 to 22 lb) on average. The mean weight of adult honey badgers from different areas has been reported at anywhere between 6.4 to 12 kg (14 to 26 lb), with a median of roughly 9 kg (20 lb), per various studies. This positions it as the third largest known badger, after the European badger and hog badger, and fourth largest extant terrestrial mustelid after additionally the wolverine. However, the average weight of three wild females from Iraq was reported as 18 kg (40 lb), about the typical weight of male wolverines or male European badgers in late autumn, indicating that they can attain much larger than typical sizes in favourable conditions. However, an adult female and two males in India were relatively small weighing 6.4 kg (14 lb) and a median of 8.4 kg (19 lb). Skull length is 13.9–14.5 cm (5.5–5.7 in) in males and 13 cm (5.1 in) for females.

# The honey badger has two pairs of mammae. It has an eversible anal pouch, a trait shared with hyenas and mongooses. The smell of the pouch is reportedly "suffocating" and may assist in calming bees when raiding beehives.
# """

# text = Hone_badger.split(".")
# sentences = [s.strip() for s in text if s.strip()]
# # print(sentences)

# # sentence transformer test
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# result = model.encode(sentences)
# print(result)

# migration
import chromadb

import json

client = chromadb.PersistentClient(path="./chroma_db")
collections = client.get_or_create_collection(name="my_collection")

data = collections.get(include=["documents", "embeddings", "metadatas"])
data_copy = data
print(id(data_copy))
data_copy["embeddings"] = [e.tolist() for e in data["embeddings"]]

with open("./data.txt", "w") as f:
    json.dump(data_copy, f)