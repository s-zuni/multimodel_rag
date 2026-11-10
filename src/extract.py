"""1단계: PDF에서 텍스트 / 표 / 이미지 추출.

실행: python src/extract.py
"""
import pickle
import time
from collections import Counter
from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

# ── 경로 설정 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = DATA_DIR / "figures"

# 본인이 고른 PDF 파일 이름 (data/ 폴더 안)
PDF_NAME = "education_test.pdf"


def extract_pdf_elements(path, fname):
    """PDF에서 이미지, 표, 텍스트 블록을 추출. 이미지는 data/figures 에 저장."""
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                 # 레이아웃 분석 모델로 정밀 추출
        extract_images_in_pdf=True,        # 그림을 별도 파일로 저장
        infer_table_structure=True,        # 표의 행·열 구조 추론
        chunking_strategy="by_title",      # 타이틀 기준으로 텍스트 분할
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=str(FIGURES_DIR),
    )


def categorize_elements(raw_pdf_elements):
    """추출된 요소를 텍스트와 표로 분류."""
    tables, texts = [], []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트를 토큰 기준 2k 단위로 재분할."""
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))


def run():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fpath = str(DATA_DIR) + "/"

    start = time.time()
    raw = extract_pdf_elements(fpath, PDF_NAME)
    print(f"추출 완료 — {time.time() - start:.1f}초, 전체 요소 {len(raw)}개")

    texts, tables = categorize_elements(raw)
    print("텍스트 블록 수:", len(texts))
    print("표 블록 수    :", len(tables))

    counts = Counter(str(type(el)).split(".")[-1].strip("'>") for el in raw)
    for name, n in counts.most_common():
        print(f"  {name:22s}: {n}개")

    images = sorted(
        p.name for p in FIGURES_DIR.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print("추출된 이미지:", len(images), "개")

    texts_2k_token = split_texts(texts)
    print("분할 후 블록 수:", len(texts_2k_token))

    out = DATA_DIR / "extracted.pkl"
    with open(out, "wb") as f:
        pickle.dump({"texts": texts, "tables": tables, "texts_2k_token": texts_2k_token}, f)
    print("저장 완료:", out)


if __name__ == "__main__":
    run()
