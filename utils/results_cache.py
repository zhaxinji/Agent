import redis
import json
import sys
sys.path.append('../')
from configs.model_config import REDIS_HOST, REDIS_PORT, REDIS_DB_CACHE, CACHE_KEY

pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_CACHE, max_connections=100)


def get_redis_connection():
    return redis.Redis(connection_pool=pool)


def get_json_from_redis():
    r = get_redis_connection()
    json_data = r.get(CACHE_KEY)
    if json_data is not None:
        return json.loads(json_data)
    return None


def store_json_to_redis(results: dict):
    # 连接到 Redis 服务器
    r = get_redis_connection()
    res = get_json_from_redis()
    print(res)
    if res:
        merged_dict = {**res, **results}
    else:
        merged_dict = {**results}
    json_data = json.dumps(merged_dict)
    r.set(CACHE_KEY, json_data)

