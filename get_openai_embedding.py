from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings # yes
import openai 
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def get_embedding_function():
    return OpenAIEmbeddings()