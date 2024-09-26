import os
from mistralai import Mistral

# Retrieve the API key from environment variables
api_key = os.environ["MISTRAL_AI_API_KEY"]

# Specify model
model = "pixtral-12b-2409"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# Define the messages for the chat
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "tag the image briefly in human language so that it can be used for semantic search and compared with user input prompt.word limit 50 words"
            },
            {
                "type": "image_url",
                "image_url": "https://drive.usercontent.google.com/download?id=1TICo_0Cp-CDh22OBFS4s-LzUecKrxNJi&authuser=0"
            }
        ]
    }
]

# Get the chat response
chat_response = client.chat.complete(
    model=model,
    messages=messages
)

# Print the content of the response
print(chat_response.choices[0].message.content)

# model = "mistral-embed"
# embeddings_batch_response = client.embeddings.create(
#     model=model,
#     inputs=["print(chat_response.choices[0].message.content)"],
# )
# print(embeddings_batch_response)