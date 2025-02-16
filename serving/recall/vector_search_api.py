from elasticsearch import Elasticsearch
import sys
import json
import argparse
from typing import List, Dict
sys.path.append('../../')
from configs.model_config import HOST, PORT, INDEX_NAME, MIN_KNN_SIMILARITY
import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def vector_search(search_text: str = "llm agent") -> List[Dict]:
    es = Elasticsearch("http://localhost:9200")

    client = OpenAI(api_key=OPENAI_API_KEY)

    query_vector = client.embeddings.create(
        input=search_text,
        model="text-embedding-3-small",
        encoding_format="float"
    ).data[0].embedding[:1024]

    search_body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'semantic_vector')",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        },
        "_source": ["item_id", "title", "authors", "date", "abstract", "body"],
        "min_score": MIN_KNN_SIMILARITY,
        "size": 1000
    }

    try:
        response = es.search(index=INDEX_NAME,
                             body=search_body)
        if response['hits']['total']['value'] > 0:
            print("Vector Search Found %d similar documents." % response['hits']['total']['value'])
            return response['hits']['hits']
        else:
            print("No similar documents found.")
            return []
    except Exception as e:
        print("Error:", e)
        return []


def knn_search(search_text="llm agent"):
    es = Elasticsearch("http://localhost:9200")

    client = OpenAI(api_key=OPENAI_API_KEY)

    query_vector = client.embeddings.create(
        input=search_text,
        model="",
        encoding_format="float"
    ).data[0].embedding

    response = es.knn_search(
        index=INDEX_NAME,
        knn={
            "field": "semantic_vector",
            "query_vector": query_vector,
            "k": 10,
            "num_candidates": 100
        },

    )

    if response['hits']['total']['value'] > 0:
        print("Found %d similar documents." % response['hits']['total']['value'])
        return response['hits']['hits']
    else:
        print("No similar documents found.")
        return []

