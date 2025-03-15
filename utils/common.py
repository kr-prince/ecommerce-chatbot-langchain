"""This module contains common utility functions used across the application."""

import os
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name: str) -> SentenceTransformer:
  """Loads the embedding model."""
  model = SentenceTransformer(
      model_name,
      token=os.getenv('HF_API_KEY', None)
  )
  return model