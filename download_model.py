import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Force HuggingFace to use a local cache directory for deployments
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./hf_cache")
model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

print(f"Downloading/Caching Hugging Face model: {model_name} to {os.environ['HF_HOME']}...")
model = SentenceTransformer(model_name)
print("Model downloaded successfully! It will now run locally from the cache during deployment.")
