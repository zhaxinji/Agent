
import gradio as gr
import sys
from pathlib import Path
import os

sys.path.append('../')
os.environ['OPENAI_API_KEY'] = ''

from configs.model_config import SEARCH_RESULTS_NUM
from serving.search_pipeline import search_api
from utils.prompt_build import build_llm_prompt
from serving.nlu.semantics_classify import semantics_classify
from serving.chat.streaming_chat import deepseek_chat
from utils.results_cache import store_json_to_redis
from serving.recall.faq_recall_api import faq_recall


def format_message(data):
    final = data[-1]
    messages = data[:-1]
    if final[0]:
        messages.append({"role": "user", "content": final[0]})
    if final[1]:
        messages.append({"role": "assistant", "content": final[1]})
    return messages

def chat(user_input, info_display, history):
    info_display.append((user_input, None))
    history.append((user_input, None))
    tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
    yield format_message(info_display), format_message(info_display), format_message(history), "", tmp_btn  # 清空输入框，立即更新界面

    prompt = user_input
    faq_res = faq_recall(str(user_input))

    if faq_res:
        info_display[-1] = (user_input, faq_res)
        history[-1] = (prompt, faq_res)
        tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
        yield format_message(info_display), format_message(info_display), format_message(history), "", tmp_btn  # 清空输入框

    else:
        class_ = semantics_classify(user_input)
        if class_ == "专业咨询":
            rerank_results = search_api(user_input, SEARCH_RESULTS_NUM)
            prompt = build_llm_prompt(user_input, rerank_results)

        history[-1] = (prompt, None)
        response_content = ""
        for partial_response in deepseek_chat(history):
            response_content = partial_response
            info_display[-1] = (user_input, response_content)
            tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
            yield format_message(info_display), format_message(info_display), format_message(history), "", tmp_btn

        info_display[-1] = (user_input, response_content)
        history[-1] = (prompt, response_content)
        tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=True)
        yield format_message(info_display), format_message(info_display), format_message(history), "", tmp_btn


def cache_agent_answer(info_display):
    if info_display:
        question, answer = info_display[-1]

        store_json_to_redis({
            question: answer
        })

    tmp_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)
    return tmp_btn

def load_css():
    css_path = Path(__file__).parent / "static" / "style.css"
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


with gr.Blocks(css=load_css(), theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="root-container"):
        with gr.Row(elem_classes="header"):
            with gr.Column(scale=8):
                gr.HTML("""
                    <div style="text-align: center;">
                        <h1 style="color: #4F46E5; font-size: 2em; margin-bottom: 8px;"></h1>
                    </div>
                """)

        with gr.Column(elem_id="main-container"):
            with gr.Column(elem_id="chatbot-container"):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    type="messages",
                    height="100%"
                )

                with gr.Row(elem_id="like-container", visible=False):
                    like_btn = gr.Button("👍 赞同", elem_id="like_button", visible=False)

            with gr.Row(elem_id="bottom-container"):
                with gr.Column(scale=4):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="请输入您的问题...",
                        elem_id="textbox",
                        container=False
                    )
                with gr.Column(scale=1, min_width=120):
                    btn = gr.Button("发送", elem_id="button")

    info_display = gr.State([])
    history = gr.State([])

    btn.click(chat,
              inputs=[txt, info_display, history],
              outputs=[chatbot, info_display, history, txt, like_btn],
              queue=True)

    txt.submit(chat,
               inputs=[txt, info_display, history],
               outputs=[chatbot, info_display, history, txt, like_btn],
               queue=True)

    like_btn.click(cache_agent_answer,
                   inputs=[info_display],
                   outputs=[like_btn],
                   queue=True)

if __name__ == "__main__":
    demo.queue().launch()