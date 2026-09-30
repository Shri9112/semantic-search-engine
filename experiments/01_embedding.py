from sentence_transformers  import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I love programming.",
    "I enjoy coading.",
    "The weather is rainy."
]

embeddings = model.encode(sentences)

for sentence,embedding in zip(sentences, embeddings):
    print("="* 60)
    print("Sentence:", sentence)
    print("Embedding:")
    print(embedding)
    print("Dimensions:", len(embedding))
