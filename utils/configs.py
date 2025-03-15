"""This module contains all the configurations and constants used in the application."""
from typing import Set

# FILE LOCATIONS
ORDERS_EXCEL_PATH: str = "./data/orders_table.xlsx"
DB_PATH: str = "./data/chatbot.db"
ENV_FILE_PATH: str = "./data/.env"
POLICY_DOCS_DIR: str = "./data/policy_docs"
POLICY_DOCS_JSON_DIR: str = "./data/policy_docs"

# DATABASE CONFIGURATIONS
ORDERS_TABLE_NAME: str = "orders"
UNIQUE_ID_COLUMN: str = "order_id"
POLICY_TABLE_NAME: str = "policy_processed"
SLEEP_TIME: int = 10

# MODEL CONFIGURATIONS
POLICY_PARSING_MODEL_NAME: str = "gemma2-9b-it"
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMS: int = 768

# PINECONE CONFIGURATIONS
PINECONE_INDEX_NAME: str = "policy-info-index"

# CHATBOT CONFIGURATIONS
SUPPORTED_INTENTS: Set = {
    "return", "refund", "exchange", "damaged item", "shipping", "payment", "replacement"
}
TOP_K: int = 50
RERANK_TOP_N: int = 5
SCORE_THRESHOLD: float = 0.0
CHATBOT_MODEL_NAME: str = "llama3-70b-8192"
CHATBOT_TEMPERATURE: float = 0.3
CHATBOT_MAX_TOKENS: int = 256





