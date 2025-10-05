# Please install dependencies first:
# pip install openai python-dotenv

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Access the API key from .env
api_key = os.getenv("DEEPSEEK_API_KEY")

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# Create a chat completion
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

# Print response
print(response.choices[0].message.content)
