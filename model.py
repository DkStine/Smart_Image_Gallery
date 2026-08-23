import os
from mistralai import Mistral
import sqlite3
import assets

api_key = os.environ["MISTRAL_AI_API_KEY"]

model = "pixtral-12b-2409"

client = Mistral(api_key=api_key)
conn = sqlite3.connect('image_descriptions.db')
cursor = conn.cursor()

for idx, image_url in enumerate(assets.image_urls):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Tag the image briefly in human language so that it can be used for semantic search, with a word limit of up to 20 words."
                },
                {
                    "type": "image_url",
                    "image_url": image_url
                }
            ]
        }
    ]
    
    try:
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )
        
        description=chat_response.choices[0].message.content
        cursor.execute('''
            INSERT OR IGNORE INTO descriptions (image_url, description)
            VALUES (?, ?)
        ''', (image_url,description))
        
        conn.commit() 
        print(f"Description for Image {idx + 1}: {description}")
    except Exception as e:
        print(f"Error processing image {idx + 1}: {e}")

conn.close()