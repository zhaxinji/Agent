import sys
import json
import argparse
sys.path.append('../../')
from sentence_transformers import SentenceTransformer
from utils.results_cache import get_json_from_redis
from configs.model_config import SIMILARITY_THRESHOLD

model = SentenceTransformer("all-MiniLM-L6-v2")


def find_max_and_index(values):
    max_value = max(values)
    index = values.index(max_value) if isinstance(values, list) else int(values.argmax())
    return max_value, index


def find_most_similar(text: str, text_list: list):
    embeddings = model.encode([text] + text_list, normalize_embeddings=True)
    scores = embeddings[0] @ embeddings[1:].T
    max_value, index = find_max_and_index(scores)
    return index, text_list[index], max_value


def faq_recall(search_text: str) -> str:
    cache_dict = get_json_from_redis()
    index, text, score = find_most_similar(str(search_text), list(cache_dict.keys()))
    if score > SIMILARITY_THRESHOLD:
        response_content = cache_dict[text]
        return response_content
    else:
        return ""



