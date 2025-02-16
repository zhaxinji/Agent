from elasticsearch import Elasticsearch
import sys
import json
import argparse
from typing import List, Dict
sys.path.append('../../')
from configs.model_config import HOST, PORT, INDEX_NAME, MIN_SEARCH_SCORE


def full_search(search_text: str) -> List[Dict]:
    es = Elasticsearch("http://localhost:9200")
    """
        •	query: 搜索的关键词或短语。
        •	fields: 一个包含多个字段名称的列表，指定在哪些字段中搜索。
        •	type: multi_match 查询的类型，支持以下几种：
        •	best_fields: 默认选项，在所有字段中查找最匹配的字段，并返回该字段的最高分数。
        •	most_fields: 在多个字段中查找最匹配的字段，并累加所有匹配字段的分数。
        •	cross_fields: 适合用于那些在不同字段中分布的查询词。
        •	phrase: 进行短语搜索。
        •	phrase_prefix: 用于实现自动完成功能。
        •	operator: or 或 and，定义匹配多个词时的逻辑操作。
    """

    query_condition = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": search_text,
                            "fields": ["title^3", "abstract^2", "body"],
                            "type": "best_fields",
                            "operator": "or"
                        }
                    }
                ]
            }
        },
        "_source": ["item_id", "title", "authors", "date", "abstract", "body"],
        "min_score": MIN_SEARCH_SCORE,
    }

    response = es.search(
        index=INDEX_NAME,
        body=query_condition,
    )

    if response['hits']['total']['value'] > 0:
        print("Fulltext Search Found %d documents." % response['hits']['total']['value'])
        return response['hits']['hits']
    else:
        print("No documents found.")
        return []

