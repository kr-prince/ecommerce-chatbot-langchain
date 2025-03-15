"""This module contains utility functions for parsing policy documents."""

import re, os
import json
import sqlite3
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_groq import ChatGroq
from textwrap import dedent

from utils.configs import (
    POLICY_PARSING_MODEL_NAME,
    POLICY_TABLE_NAME
)

def split_text_into_subsections(text):
    """
    Splits text into subsections using MarkdownHeaderTextSplitter.
    """
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("**", "Header 4")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sub_sections_pattern = re.compile(r"\*\*\d+\.\s")
    document_headers = markdown_splitter.split_text(text)
    document_subsections = []
    for doc_head in document_headers:
        sub_sections = sub_sections_pattern.split(doc_head.page_content)
        sub_sections = [ss.strip() for ss in sub_sections if ss.strip()]
        document_subsections.extend(sub_sections)
    return document_subsections

def process_subsections_with_llm(document_subsections):
    """ Given a document subsection return the intents and summary """
    chat = ChatGroq(
        model_name=POLICY_PARSING_MODEL_NAME,
        temperature=0.1,
        api_key=os.getenv('GROQ_API_KEY'),
        max_tokens=2048
      )
    messages = [
        {
            "role": "system",
            "content": dedent("""
            Think like a good customer service agent in E-commerce business and follow the below instructions as it is:
              1. Extract the main intent covered in the text in one or two words. For ex - refund, replacement, etc.
                Add multiple intents as applicable for the text, but not more than top 5.
              2. Summarize the text into a list of very short sentences without loosing any critical info. Do not repeat same sentences.
              3. Provide the collated output strictly in JSON format as shown below. Just give the output without any extra text or explanation.
              {
                "intents": ["intent 1", "intent 2", ...],
                "summary": ["short sentence 1", "sentence 2", ...]
              }
            """)
        }
    ]
    json_responses = []
    for document_subsection in document_subsections:
        current_messages = messages + [{"role": "user", "content": document_subsection}]
        try:
            response = chat.invoke(current_messages)
            json_responses.append(json.loads(response.content))
        except Exception as e:
            print(f"Error processing subsection with LLM: {e}")
    return json_responses

def save_processed_filename(filename, db_path, json_responses, json_output_dir):
    """ Save the json file and processed filename to the database """
    try:
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_file_path = os.path.join(json_output_dir, json_filename)
        with open(json_file_path, 'w') as f:
            json.dump(json_responses, f, indent=2)
        print(f"Saved JSON file: {json_file_path}")

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"INSERT OR IGNORE INTO {POLICY_TABLE_NAME} (filename, status) VALUES (?, ?)", (filename, 'parsing done'))
            conn.commit()
        print(f"Saved processed filename: {filename}")
    except Exception as e:
        print(f"Error saving processed filename: {e}")

def check_file_processed(filename, db_path):
    """ Check if a file has already been processed """
    try:
        flag = None
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''CREATE TABLE IF NOT EXISTS {POLICY_TABLE_NAME} (filename TEXT UNIQUE, status TEXT)''')
            cursor.execute(f"SELECT 1 FROM {POLICY_TABLE_NAME} WHERE filename = ?", (filename,))
            if cursor.fetchone():
                flag = True
            else:
                flag = False
    except Exception as e:
        print(f"Error checking file processed: {e}")
        flag = False
    return flag

def process_pdf_file(pdf_file_path, db_path, json_output_dir):
    """ Process a PDF file and return the processed
    data in JSON format """
    filename = os.path.basename(pdf_file_path)
    text = pymupdf4llm.to_markdown(pdf_file_path)
    if text:
        document_subsections = split_text_into_subsections(text)
        json_responses = process_subsections_with_llm(document_subsections)
        save_processed_filename(filename, db_path, json_responses, json_output_dir)
    else:
        print(f"No text extracted from file: {filename}")
        json_responses = []
    return json_responses

