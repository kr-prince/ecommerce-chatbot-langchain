"""This module contains utility functions for ingesting policy documents."""

import os
import time
import sqlite3
import hashlib
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from utils.configs import (
   POLICY_TABLE_NAME
)

def generate_id_for_text(text: str) -> str:
    """Generates a deterministic ID for a given text."""
    return hashlib.sha256(text.encode()).hexdigest()

def collate_json_data(data: List[Dict]) -> List[Dict]:
    """Collates data from a list of JSON objects."""
    collated_data = {}
    for item in data:
        for text in item['summary']:
            text = text.lower()
            text_id = generate_id_for_text(text)
            if text_id not in collated_data:
                collated_data[text_id] = {
                    'id': text_id,
                    'text': text,
                    'intents': []
                }
            intents = list(set(map(str.lower, item['intents'])))
            collated_data[text_id]['intents'].extend(intents)
    return list(collated_data.values())

def create_or_load_pinecone_index(index_name: str, embedding_dims: int) -> Pinecone.Index:
    """Creates or loads a Pinecone index."""
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=embedding_dims,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print(f"Index {index_name} is ready")
    else:
        print(f"Index {index_name} already exists")
    return pc.Index(index_name)

def process_and_upsert_data(index: Pinecone.Index, collated_data: List[Dict], model: SentenceTransformer) -> None:
    """Processes and upserts data to Pinecone index."""
    inserted_count = 0
    upserted_count = 0
    for record in collated_data:
        try:
            response = index.fetch([record['id']])
            if record['id'] in response.vectors:
                existing_metadata = response.vectors[record['id']]['metadata']
                existing_metadata['intents'] = list(set(existing_metadata['intents']+record['intents']))
                index.upsert(vectors=[{
                    'id': record['id'],
                    'values': model.encode(record['text']).tolist(),
                    'metadata': existing_metadata
                }])
                upserted_count += 1
            else:
                index.upsert(vectors=[{
                    'id': record['id'],
                    'values': model.encode(record['text']).tolist(),
                    'metadata': {
                        'text': record['text'],
                        'intents': record['intents']
                    }
                }])
                inserted_count += 1
        except Exception as e:
            print(f"Error processing record {record['id']}: {e}")
    print(f"Inserted {inserted_count} records, Upserted {upserted_count} records")
    assert inserted_count + upserted_count == len(collated_data), "Inserted + upserted do not match total data"

def update_database_status(db_path: str, filename: str, status: str) -> None:
    """Updates the database status."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE {POLICY_TABLE_NAME} SET status = ? WHERE filename = ?", (status, filename))
        conn.commit()

def get_files_to_process(db_path: str) -> List[str]:
    """Returns a list of filenames with status 'parsing done'."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT filename FROM {POLICY_TABLE_NAME} WHERE status = ?", ('parsing done',))
        filenames = [row[0] for row in cursor.fetchall()]
    return filenames