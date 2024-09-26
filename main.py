from sentence_transformers import SentenceTransformer
import numpy as np

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Texts to compare
text1 = "A sunny beach with clear skies and soft sand."
text2 = "A beach with clear weather, bright sunshine, and sandy shore"

# Get embeddings
embedding1 = model.encode(text1)
embedding2 = model.encode(text2)

# Cosine Similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

similarity = cosine_similarity(embedding1, embedding2)
print(f'Cosine Similarity: {similarity}')