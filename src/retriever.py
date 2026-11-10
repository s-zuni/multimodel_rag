"""3단계: 다중 벡터 검색기 + 멀티모달 체인 + 텍스트 전용 비교 체인.

실행: python src/retriever.py   (질의응답 5건 + Before/After)
"""
import os
import pickle
import re
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── 경로 설정 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")

ID_KEY = "doc_id"
B64_PATTERN = r"^[A-Za-z0-9+/]{100,}={0,2}$"
INSTRUCTION = ("아래 자료를 근거로 질문에 한국어로 답하세요.\n"
               "자료에 없는 내용은 모른다고 답하세요.\n\n")

QUESTIONS = [
    # Q1 (그래프): 출생월별 부의 교육년수 막대그래프 — 수치가 그래프에만 있음 (Before/After 비교용)
    "출생월별 아버지의 교육년수 막대그래프에서 막대 높이는 대략 몇 년인가요?",
    # Q2 (그래프): 출생월별 출생아 수 막대그래프
    "출생월별 출생아 수 막대그래프에서 막대가 가장 높은 출생월은 몇 월이고 대략 몇 명인가요?",
    # Q3 (표): 학년별 국어/수학/영어 OLS vs IV 추정 결과
    "4학년의 국어 성적에 대한 OLS 추정계수와 IV 추정계수는 각각 얼마인가요?",
    # Q4 (표): 기초통계량
    "표본의 평균 실제 취학나이와 의무 취학나이, 형제 수 평균은 얼마인가요?",
    # Q5 (텍스트): 연구 방법과 결론
    "이 연구는 취학 나이의 내생성 문제를 어떻게 해결했고, 어떤 결론을 얻었나요?",
]
COMPARE_QUERY = QUESTIONS[0]  # Before/After 비교에 쓸 질문: 그래프에만 있는 정보


def check_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("OPENAI_API_KEY가 없습니다. .env 파일을 확인해주세요.")


def load_processed():
    with open(DATA_DIR / "extracted.pkl", "rb") as f:
        ext = pickle.load(f)
    with open(DATA_DIR / "summaries.pkl", "rb") as f:
        summ = pickle.load(f)
    return ext, summ


def build_retriever():
    """요약은 vectorstore(검색용), 원본은 docstore(전달용)에 같은 id로 저장."""
    ext, summ = load_processed()
    vectorstore = Chroma(
        collection_name="multimodal_rag",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=InMemoryStore(), id_key=ID_KEY
    )

    def add_documents(doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)               # 요약 → 검색용
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))      # 원본 → 전달용

    if summ["text_summaries"]:
        add_documents(summ["text_summaries"], ext["texts_2k_token"])
    if summ["table_summaries"]:
        add_documents(summ["table_summaries"], ext["tables"])
    if summ["image_summaries"]:
        add_documents(summ["image_summaries"], summ["img_base64_list"])
    print("색인 완료 — 벡터 DB 요약 수:", len(vectorstore.get()["ids"]))
    return retriever, ext


def is_image(content):
    return bool(re.match(B64_PATTERN, content))


def doc_kind(doc):
    content = doc.page_content if hasattr(doc, "page_content") else doc
    return "이미지" if is_image(content) else "텍스트/표"


def split_image_text_types(docs):
    b64_images, texts_out = [], []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        (b64_images if is_image(content) else texts_out).append(content)
    return {"images": b64_images, "texts": texts_out}


def img_prompt_func(data_dict):
    messages = []
    for image in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
        "type": "text",
        "text": INSTRUCTION + f"질문: {data_dict['question']}\n\n참고 텍스트:\n{formatted_texts}",
    })
    return [HumanMessage(content=messages)]


def build_model():
    return ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)


def build_multimodal_chain(retriever, model):
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )


def build_text_only_chain(texts_2k_token, model):
    """비교용: 이미지·표를 뺀 것 말고는 동일한 조건."""
    store = Chroma(
        collection_name="text_only_rag",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    store.add_texts(texts_2k_token)
    text_retriever = store.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template(
        INSTRUCTION + "질문: {question}\n\n참고 텍스트:\n{context}"
    )

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    return (
        {"context": text_retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


def run():
    check_api_key()
    retriever, ext = build_retriever()
    model = build_model()
    chain = build_multimodal_chain(retriever, model)

    for i, q in enumerate(QUESTIONS, 1):
        kinds = [doc_kind(d) for d in retriever.invoke(q)]
        start = time.time()
        answer = chain.invoke(q)
        print("=" * 70)
        print(f"[Q{i}] {q}  ({time.time() - start:.1f}초)")
        print("근거 원본:", kinds)
        print("-" * 70)
        print(answer)

    text_chain = build_text_only_chain(ext["texts_2k_token"], model)
    print("\n" + "=" * 70)
    print("Q:", COMPARE_QUERY)
    print("=" * 70)
    print("\n[BEFORE] 텍스트 전용 RAG\n" + "-" * 70)
    print(text_chain.invoke(COMPARE_QUERY))
    print("\n[AFTER] 멀티모달 RAG\n" + "-" * 70)
    print(chain.invoke(COMPARE_QUERY))


if __name__ == "__main__":
    run()
