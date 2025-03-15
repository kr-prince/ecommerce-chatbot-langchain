"""This is the main file for the Policy Ingestion Service. It is responsible for
ingesting policies from a given source and storing them in the database every hour."""

from utils.policy_parsing import (
    check_file_processed,
    process_pdf_file,
)
from utils.policy_ingestion import (
    get_files_to_process,
    update_database_status,
    collate_json_data,
    create_or_load_pinecone_index,
    process_and_upsert_data
)
from utils.configs import (
    POLICY_DOCS_DIR,
    POLICY_DOCS_JSON_DIR,
    DB_PATH,
    SLEEP_TIME,
    ENV_FILE_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMS,
    PINECONE_INDEX_NAME
)
from utils.common import load_embedding_model

import os
import json
import time
from dotenv import load_dotenv

def parse_policy_documents(input_dir, db_path, json_output_dir):
  """Parses policy documents from the input directory and saves the JSON output."""
  for filename in os.listdir(input_dir):
      if filename.endswith(".pdf"):
          file_processed_flag = check_file_processed(filename, db_path)
          if not file_processed_flag:
              print(f"Processing file: {filename}")
              pdf_file_path = os.path.join(input_dir, filename)
              json_responses = process_pdf_file(pdf_file_path, db_path, json_output_dir)
              if len(json_responses) > 0:
                  print(f"Processed file: {filename}, {len(json_responses)} JSON record(s) saved.")

def upload_policy_to_pinecone(input_dir, db_path):
    """Uploads JSON policy data to Pinecone index."""
    filenames = get_files_to_process(db_path)
    for filename in filenames:
        json_file = os.path.splitext(filename)[0] + '.json'
        json_file_path = os.path.join(input_dir, json_file)

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            collated_data = collate_json_data(data)
            index = create_or_load_pinecone_index(PINECONE_INDEX_NAME, EMBEDDING_DIMS)
            model = load_embedding_model(EMBEDDING_MODEL_NAME)
            process_and_upsert_data(index, collated_data, model)
            update_database_status(db_path, filename, 'index updated')
  
if __name__ == "__main__":
  load_dotenv(ENV_FILE_PATH)
  while True:
      parse_policy_documents(POLICY_DOCS_DIR, DB_PATH, POLICY_DOCS_JSON_DIR)
      upload_policy_to_pinecone(POLICY_DOCS_JSON_DIR, DB_PATH)
      time.sleep(SLEEP_TIME)
