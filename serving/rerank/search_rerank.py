import sys
import argparse
import json
from typing import List, Dict
sys.path.append('../../')
from configs.model_config import MODEL_PATH
from FlagEmbedding import FlagReranker
from serving.recall.full_text_search_api import full_search
from serving.recall.vector_search_api import vector_search

MODEL_LIST = MODEL_PATH["reranker"]


def rerank_search_results(search_text: str, full_search_recall: List[Dict], vec_recall: List[Dict], keep_num: int = 10) -> List[Dict]:

    full_search_recall_info = []
    for e in full_search_recall:
        e['method'] = 'full_search'
        full_search_recall_info.append(e)

    vec_recall_info = []
    for e in vec_recall:
        e['method'] = 'vector_search'
        vec_recall_info.append(e)

    recalls = full_search_recall_info + vec_recall_info

    unique_items = {}
    for item in recalls:
        item_id = item['_source']['item_id']
        if item_id not in unique_items:
            unique_items[item_id] = item

    recalls = list(unique_items.values())

    pairs = []
    for i in range(len(recalls)):
        recall = recalls[i]
        title = recall["_source"]["title"]
        authors = recall["_source"]["authors"]
        abstract = recall["_source"]["abstract"]
        body = recall["_source"]["body"]

        doc_info = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "body": body,
        }
        pairs.append([search_text, str(doc_info)])

    reranker = FlagReranker(MODEL_LIST["bge-reranker-v2-m3"], use_fp16=True)  # Setting use_fp16 to True speeds up

    if pairs:
        scores = reranker.compute_score(pairs, normalize=True)

        combined_list = list(zip(recalls, scores))

        sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

        result_list = []
        for x in sorted_combined_list:
            temp = x[0]
            temp['rerank_score'] = x[1]
            result_list.append(temp)
        return result_list[:keep_num]
    else:
        return []

