import fitz
import re
import sys
import json
from openai import OpenAI
import os
sys.path.append('../../')
from configs.model_config import (DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, AUTHORS_INFO_PROMPT,
                                                ABSTRACT_PROMPT, REFERENCES_PROMPT, APPENDIX_PROMPT)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def extract_date_and_title(file_name):
    date_pattern = r"\d{4}(?:\.\d{2})?(?:\.\d{2})?"
    date_match = re.search(date_pattern, file_name)
    date = date_match.group() if date_match else "日期未找到"
    title_start = file_name.find(date) + len(date) + 1
    title = file_name[title_start:].replace(".pdf", "").strip()

    return date, title


def get_authors(text: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system",
             "content": "你是一个文本处理专家，基于给定的文本信息提取关键信息。"},
            {"role": "user", "content": AUTHORS_INFO_PROMPT.format(TEXT=text[:len(text) // 2])}
        ],
        response_format={
            'type': 'json_object'
        }
    )
    result = completion.choices[0].message.content
    data_dict = json.loads(result)

    return data_dict


def get_abstract(text: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system",
             "content": "你是一个文本处理专家，基于给定的文本信息提取关键信息。"},
            {"role": "user", "content": ABSTRACT_PROMPT.format(TEXT=text[:len(text) // 2])}
        ],
    )
    result = completion.choices[0].message.content

    return result


def find_and_concatenate(lst, keyword='REFERENCES\n'):
    for i, element in enumerate(lst):
        if keyword.lower() in element.lower():
            result = ' '.join(lst[i:])
            return result
    return ""


def load_partial_json(result):
    try:
        data_dict = json.loads(result)
        return data_dict
    except json.JSONDecodeError:
        match = re.search(r'("references": \[.*?)(,[^,]*?)?\s*$', result, re.DOTALL)
        if match:
            corrected_result = "{" + match.group(1) + "\"]}"
            try:
                data_dict = json.loads(corrected_result)
                return data_dict
            except json.JSONDecodeError as e:
                print(f"仍然无法解析 JSON：{e}")
                return {}
        else:
            print("找不到 'references' 列表")
            return {}
def get_references(text: str):


    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system",
             "content": "你是一个文本处理专家，基于给定的文本信息提取关键信息。"},
            {"role": "user", "content": REFERENCES_PROMPT.format(TEXT=text[:len(text) // 2])}
        ],
        response_format={
            'type': 'json_object'
        },
    )
    result = completion.choices[0].message.content
    data_dict = load_partial_json(result)

    return data_dict


def get_appendix(text: str):
    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system",
             "content": "你是一个文本处理专家，基于给定的文本信息提取关键信息。"},
            {"role": "user", "content": APPENDIX_PROMPT.format(TEXT=text[:len(text) // 2])}
        ],
    )
    result = completion.choices[0].message.content

    return result


def extract_between_strings_case_insensitive(A, a, b):
    A_lower = A.lower()
    a_lower = a.lower()
    b_lower = b.lower()

    start = A_lower.find(a_lower)
    end = A_lower.find(b_lower, start + len(a_lower))

    if start == -1 or end == -1:
        return ""

    return A[start + len(a):end]


def extract_after_strings_case_insensitive(A, a):
    A_lower = A.lower()
    a_lower = a.lower()

    start = A_lower.find(a_lower)

    if start == -1:
        return ""

    return A[start + len(a):]


def extract_pdf_to_json(pdf_path):
    date, title = extract_date_and_title(pdf_path)

    structured_data = {
        "title": title,
        "authors": None,
        "date": date,
        "abstract": None,
        "body": None,
        "references": None,
        "appendix": None
    }

    document = fitz.open(pdf_path)

    page_0_text = document.load_page(0).get_text("text")

    authors = get_authors(page_0_text)
    abstract = get_abstract(page_0_text)

    structured_data['authors'] = authors
    structured_data['abstract'] = abstract

    content = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text = page.get_text("text")
        content.append(text)

    concatenated_string = find_and_concatenate(content)
    references = get_references(concatenated_string)
    appendix = ""
    if references:
        if 'references' in references:
            references = references['references']
            if references:
                append_start = references[-1][-10:]
                appendix = extract_after_strings_case_insensitive(concatenated_string, append_start)

    structured_data['references'] = references
    structured_data['appendix'] = appendix

    full_content = ' '.join(content)

    body = extract_between_strings_case_insensitive(full_content, abstract[-10:], '\nREFERENCES\n')
    structured_data['body'] = body

    return structured_data


def save_text_to_file(text, file_name="output.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Text successfully written to {file_name}")



