import sys
import json
sys.path.append('../../')
from elasticsearch import Elasticsearch, helpers
from configs.model_config import HOST, PORT, INDEX_NAME
from data_preprocess.structure_pdf.paper_structure import extract_pdf_to_json
from utils.little_tools import get_md5, list_files_with_os
from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_index():
    es = Elasticsearch("http://localhost:9200")

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "item_id": {
                    "type": "keyword"
                },
                "title": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "authors": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "text"},
                        "work": {"type": "text"},
                        "contact": {"type": "text"}
                    }
                },
                "date": {
                    "type": "date",
                    "format": "yyyy-MM-dd"
                },
                "abstract": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "body": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "`references`": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "appendix": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "semantic_vector": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    # 创建索引
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=mapping)
        print(f"索引 '{INDEX_NAME}' 已创建。")
    else:
        print(f"索引 '{INDEX_NAME}' 已存在。")



def insert_into_es(documents: list):
    es = Elasticsearch("http://localhost:9200")

    actions = [
        {
            "_index": INDEX_NAME,
            "_id": doc["item_id"],  # 可以根据需要使用自定义ID
            "_source": doc
        }
        for doc in documents
    ]

    try:
        helpers.bulk(es, actions)
        print(f"{len(documents)} 个文档已插入到索引 '{INDEX_NAME}' 中。")
    except helpers.BulkIndexError as e:
        for error in e.errors:
            print(json.dumps(error, indent=2, ensure_ascii=False))

        for error in e.errors:
            for op_type, details in error.items():
                if not details.get("status") == 200:
                    print(f"Error in document ID {details['_id']}: {details['error']['reason']}")


def load_paper_2_es(paper_info: dict):

    documents = []
    abstract = str(paper_info['abstract'])

    client = OpenAI(api_key=OPENAI_API_KEY)

    semantic_vector = client.embeddings.create(
        input=abstract,
        model="",
        encoding_format="float"
    ).data[0].embedding

    document = {
        "item_id": get_md5(paper_info['title']),
        "title": paper_info['title'],
        "authors": paper_info['authors'],
        "date": paper_info['date'].replace('.', '-'),
        "abstract": paper_info['abstract'],
        "body": paper_info['body'],
        "references": paper_info['references'],
        "appendix": paper_info['appendix'],
        "semantic_vector": semantic_vector
    }
    documents.append(document)
    insert_into_es(documents)


def load_all_papers_2_es(path: str):
    files = list_files_with_os(path)
    index = 0
    for file in files:
        index += 1
        info = extract_pdf_to_json(file)
        output_info = {key: info[key] for key in ['item_id', 'title', 'authors', 'date', 'abstract'] if key in info}
        pretty_result = json.dumps(output_info, indent=4, ensure_ascii=False)
        load_paper_2_es(info)
        break
