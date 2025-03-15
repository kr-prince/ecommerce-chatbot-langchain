"""This module contains all the functions which the chat agent uses as tools."""

from utils.configs import (
    SUPPORTED_INTENTS,
    DB_PATH,
    ORDERS_TABLE_NAME,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMS,
    TOP_K, RERANK_TOP_N, SCORE_THRESHOLD,
    ENV_FILE_PATH
)
from utils.policy_ingestion import (
    create_or_load_pinecone_index
)
from utils.common import load_embedding_model
from typing import List

import os
import random
from datetime import datetime
from typing import List
import sqlite3
from langchain_core.tools import ToolException
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv(ENV_FILE_PATH)
pc = Pinecone(os.getenv("PINECONE_API_KEY"))
pc_index = create_or_load_pinecone_index(
    index_name=PINECONE_INDEX_NAME,
    embedding_dims=EMBEDDING_DIMS
)
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)

def get_intents_from_query(query_text: str) -> List[str]:
    """Retrieves a list of intents from the SUPPORTED INTENTS from the given text."""
    intents = []
    for intent in SUPPORTED_INTENTS:
        if intent in query_text.lower():
            intents.append(intent)
    return intents

def get_similar_products_for_order(order_id: int) -> List[str]:
    """
    Retrieves a list of similar products by given order ID. Use this tool only if the user
    asks for a product recommendation. It checks for the same product category,
    size and gender(or unisex).

    Args:
        order_id: The 5-digit order ID.

    Returns:
        A list of similar product names.

    Raises:
        ToolException: If no similar products are found or order_id is invalid.
    """
    try:
        similar_products = []
        if (not isinstance(order_id, int)) or (not (10000 <= order_id <= 99999)):
          raise ToolException("Order ID must be a five-digit number.")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Get product category, gender, and size for the given order ID
            cursor.execute(f"""
                SELECT product_category, gender, size
                FROM {ORDERS_TABLE_NAME}
                WHERE order_id = ?
            """, (order_id,))

            result = cursor.fetchone()

            if result:
                product_category, gender, size = result

                # Find similar products based on gender, category, and size
                cursor.execute(f"""
                    SELECT product_name
                    FROM {ORDERS_TABLE_NAME}
                    WHERE ((gender = ? OR lower(gender) = 'unisex')
                        AND product_category = ?
                        AND size = ?
                        AND order_id != ? )
                    ORDER BY quantity DESC
                    LIMIT 3
                """, (gender, product_category, size, order_id))

                similar_products = [row[0] for row in cursor.fetchall()]

        if len(similar_products) == 0:
            raise ToolException("No similar products found for this order ID.")
    except Exception as ex:
        raise ToolException(str(ex))
    return similar_products

def generate_return_authorization(order_id: int) -> str:
    """Generates a return authorization number for the given order ID if user asks to return an order

    Args:
        order_id: The five-digit order ID.

    Returns:
        The generated return authorization number.

    Raises:
        ToolException: If the order ID is not a five-digit number.
    """
    if not (10000 <= order_id <= 99999):
        raise ToolException("Order ID must be a five-digit number.")

    # Use the order ID to seed the random number generator for deterministic behavior
    random.seed(order_id)
    random_number = random.randint(10000, 99999)
    return f"RA{random_number}"

def get_order_details(order_id: int) -> dict:
    """
    Fetches order details from the database given the order ID.

    Args:
        order_id: The 5-digit order ID of the order to retrieve.

    Returns:
        A dictionary containing the order details as key values.

    Raises:
        ToolException: If no order is found with the given order_id or order_id is invalid.
    """
    try:
        if (not isinstance(order_id, int)) or (not (10000 <= order_id <= 99999)):
            raise ToolException("Order ID must be a five-digit number.")
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT
                    order_id, product_category, product_name, size, quantity, `price_(usd)`,
                    order_date, status, payment_method, shipping_address, final_sale
                FROM {ORDERS_TABLE_NAME}
                WHERE order_id = ?
            """, (order_id,))

            result = cursor.fetchone()

            if result is None:
                raise ToolException(f"No order found with ID: {order_id}")

            order_details = {
                "order_id": result[0],
                "product_category": result[1],
                "product_name": result[2],
                "size": result[3],
                "quantity": result[4],
                "price_(usd)": result[5],
                "order_date": datetime.strptime(result[6], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d'),  # Format date
                "status": result[7],
                "payment_method": result[8],
                "shipping_address": result[9],
                "final_sale": result[10]
            }
            
            return order_details

    except sqlite3.Error as ex:
        raise ToolException(f"Database error: {ex}")
    except Exception as ex:
        raise ToolException(str(ex))

def retrieve_relevant_policies_by_query(query_text: str) -> List[str]:
    """Retrieves the most relevant policies from the vector store for the given user query.
    The policies are ranked as per their relevance with the given user query.

    Args:
        query_text: The user's query string.

    Returns:
        A list of the most relevant policies for the given query.

    Raises:
        ToolException: If any error occurs during the process.
    """
    try:
        # Search Pinecone index
        query_intents = get_intents_from_query(query_text)
        query_response = pc_index.query(
            vector=embedding_model.encode(query_text).tolist(),
            top_k=TOP_K,
            include_metadata=True,
            include_values=False,
            filter={
                "intents": {
                    "$in": query_intents
                }
            } if len(query_intents) > 0 else None
        )

        # Prepare policies for reranking
        policies = [
            {"id": x["id"], "text": x["metadata"]["text"]}
            for x in query_response["matches"]
        ]

        # Rerank policies
        reranked_policies = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query_text,
            documents=policies,
            top_n=RERANK_TOP_N,
            return_documents=True,
        )

        # Filter out low-scoring policies
        filtered_policies = [
            doc.document.text for doc in reranked_policies.rerank_result.data \
                if doc["score"] >= SCORE_THRESHOLD
        ]
        
        return filtered_policies

    except Exception as ex:
        raise ToolException(str(ex))

def days_since_date(date_str: str) -> int:
    """Calculates the number of days passed since the given order date.

    Args:
        date_str: The date string in YYYY-MM-DD format.

    Returns:
        The number of days passed since the given order date.

    Raises:
        ToolException: If the date string is not in the correct format or if there's an error during date processing.
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        today = datetime.now()
        delta = today - date_obj
        return delta.days
    except ValueError:
        raise ToolException("Incorrect date format. Please use YYYY-MM-DD.")
    except Exception as e:
        raise ToolException(f"An error occurred: {e}")

