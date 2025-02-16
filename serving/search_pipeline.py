import sys
from fastapi import Query
import argparse
import json
sys.path.append('../')
from serving.recall.full_text_search_api import full_search
from serving.recall.vector_search_api import vector_search
from serving.rerank.search_rerank import rerank_search_results


def search_api(search_text: str = Query(..., description="搜索词"),
               keep_num: int = Query(1000, description="返回结果数量")):
    full_search_recall = full_search(search_text)
    vec_recall = vector_search(search_text)
    rerank_results = rerank_search_results(search_text, full_search_recall, vec_recall, keep_num=keep_num)
    return rerank_results


