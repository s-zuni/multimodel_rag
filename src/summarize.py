"""2단계: 텍스트 / 표 / 이미지 요약 생성.

실행: python src/summarize.py
"""
import base64
import os
import pickle
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ── 경로 설정 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = DATA_DIR / "figures"

load_dotenv(PROJECT_ROOT / ".env")

PROMPT_TEXT_KOR = """
"표와 텍스트를 검색용으로 요약하는 도우미 / 임베딩되어 원본 검색에 쓰임 /
 검색에 최적화된 간결한 요약 작성. 원본텍스트 수의 15%정도 글자수로 요약해 / 요약할 표 또는 텍스트: {element}"""

# 지시문은 영어, 결과는 한국어로 (토큰 효율 + 품질)
PROMPT_IMG = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""


def check_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("OPENAI_API_KEY가 없습니다. .env 파일을 확인해주세요.")


def generate_text_summaries(texts, tables, summarize_texts=False):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEXT_KOR)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries, table_summaries = [], []
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 2})
    elif texts:
        text_summaries = texts
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})
    return text_summaries, table_summaries


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
        ])
    ])
    return msg.content


def generate_img_summaries(path):
    img_base64_list, image_summaries = [], []
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            b64 = encode_image(os.path.join(path, img_file))
            img_base64_list.append(b64)
            image_summaries.append(image_summarize(b64, PROMPT_IMG))
    return img_base64_list, image_summaries


def run():
    check_api_key()
    with open(DATA_DIR / "extracted.pkl", "rb") as f:
        data = pickle.load(f)
    texts_2k_token, tables = data["texts_2k_token"], data["tables"]
    print(f"texts_2k_token {len(texts_2k_token)}개 / tables {len(tables)}개")

    start = time.time()
    text_summaries, table_summaries = generate_text_summaries(
        texts_2k_token, tables, summarize_texts=True
    )
    print(f"텍스트·표 요약 완료 — {time.time() - start:.1f}초 "
          f"(텍스트 {len(text_summaries)}, 표 {len(table_summaries)})")

    start = time.time()
    img_base64_list, image_summaries = generate_img_summaries(str(FIGURES_DIR))
    print(f"이미지 요약 완료 — {time.time() - start:.1f}초 ({len(image_summaries)}장)")

    if text_summaries:
        print("\n[텍스트 요약 예시]\n", text_summaries[0])
    if image_summaries:
        print("\n[이미지 요약 예시]\n", image_summaries[0])

    out = DATA_DIR / "summaries.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "text_summaries": text_summaries,
            "table_summaries": table_summaries,
            "image_summaries": image_summaries,
            "img_base64_list": img_base64_list,
        }, f)
    print("저장 완료:", out)


if __name__ == "__main__":
    run()
