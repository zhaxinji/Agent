import sys
from openai import OpenAI
sys.path.append('../../')
import os
import time
from configs.model_config import (DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def deepseek_chat(history):
    messages = [{"role": "system", "content": "你是一个有帮助的人工智能助手，基于给定的Prompt来回答问题。"}]
    for human_message, assistant_message in history:
        if human_message:
            messages.append({"role": "user", "content": human_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})

    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )

    response_content = ""
    for chunk in response:
        if hasattr(chunk, "choices") and chunk.choices[0].delta:
            content = chunk.choices[0].delta.content
            if content is not None:
                response_content += content
            yield response_content

