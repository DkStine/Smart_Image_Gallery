import sqlite3
from sentence_transformers import SentenceTransformer
import assets

model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------send to db

conn = sqlite3.connect("image_descriptions.db")
cursor = conn.cursor()

user_input = input(f"Input your query: ")
user_input = model.encode(user_input) 


cursor.execute("SELECT *  FROM descriptions")
rows = cursor.fetchall()

sample_dict = {}

for row in rows:
    sample_dict[row[1]] = model.encode(row[2])

for img_url in sample_dict.keys():
    similarity = assets.cosine_similarity(user_input, sample_dict[img_url])
    if similarity > 0.5:
        print(f"Image url: {img_url}")

conn.close()