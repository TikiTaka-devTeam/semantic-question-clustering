import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_FILE = "2-3)_2장._OSI_모델과_TCP-IP_프로토콜_(31_pages).pdf"
CATEGORY_FILE = "categories.json"
STORE_FILE = "question_store.json"

EMBEDDING_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-5.4-mini"
SIM_THRESHOLD = 0.80
AMBIGUOUS_LOW = 0.70
LONG_QUESTION_LEN = 80


# 공통 유틸
def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)

    return "\n".join(texts).strip()


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(text: str):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding


def call_json_response(prompt: str) -> dict:
    resp = client.responses.create(
        model=GEN_MODEL,
        input=prompt
    )
    text = resp.output_text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


# 1단계: PDF 기반 카테고리 생성
def generate_categories_from_material(material_text: str) -> list[dict]:
    prompt = f"""
너는 강의자료 분석 도우미다.
아래 강의자료를 보고, 학생 질문 분류에 적합한 카테고리를 생성하라.

조건:
- 카테고리는 5~10개
- 너무 겹치지 않게 만들 것
- 마지막 카테고리는 무조건 "기타"로 할 것
- 각 카테고리마다:
  1) id
  2) name
  3) description
  4) keywords (3~6개)
를 포함할 것

반드시 아래 JSON 형식으로만 답하라.
{{
  "categories": [
    {{
      "id": "cat1",
      "name": "예시",
      "description": "무엇을 다루는지",
      "keywords": ["키워드1", "키워드2"]
    }}
  ]
}}

강의자료:
{material_text}
"""
    result = call_json_response(prompt)
    return result["categories"]


def initialize_categories():
    if os.path.exists(CATEGORY_FILE):
        print("기존 categories.json 사용")
        return load_json(CATEGORY_FILE, [])

    material_text = load_pdf_text(PDF_FILE)
    categories = generate_categories_from_material(material_text)
    save_json(CATEGORY_FILE, categories)
    print("카테고리 생성 완료")
    return categories


# 2단계: 질문 카테고리 분류
def classify_question(question: str, categories: list[dict]) -> dict:
    category_text = "\n".join(
        [
            f"- {c['id']} | {c['name']} | {c['description']} | keywords={', '.join(c['keywords'])}"
            for c in categories
        ]
    )

    prompt = f"""
너는 학생 질문을 강의 카테고리에 분류하는 분류기다.

아래 카테고리 목록 중 가장 적절한 카테고리 하나를 고르고,
confidence를 0~1 사이 실수로 추정하라.

질문이 너무 장황하거나, 두 개 이상의 의도가 섞였거나, 카테고리가 애매하면
needs_refine을 true로 하라.

반드시 아래 JSON 형식으로만 답하라.
{{
  "category_id": "cat1",
  "confidence": 0.91,
  "needs_refine": false,
  "reason": "짧게"
}}

카테고리:
{category_text}

질문:
{question}
"""
    return call_json_response(prompt)


# 4단계: 애매/장황 질문만 추가 정제
def refine_question(question: str, categories: list[dict]) -> dict:
    category_names = ", ".join([c["name"] for c in categories])

    prompt = f"""
너는 학생 질문 정제 도우미다.
아래 장황하거나 애매한 질문을,
핵심 의미를 유지하면서 짧고 비교하기 쉬운 질문 한 문장으로 바꿔라.

조건:
- 인사말, 군더더기 제거
- 질문 의도가 여러 개면 가장 핵심 질문 하나만 남김
- 가능한 한 아래 카테고리들 중 하나에 잘 대응되도록 정리
- 원래 의미는 유지

카테고리 후보:
{category_names}

반드시 아래 JSON 형식으로만 답하라.
{{
  "refined_question": "정제된 질문",
  "reason": "짧게"
}}

원문 질문:
{question}
"""
    return call_json_response(prompt)


# 저장소 관리
def load_store():
    return load_json(STORE_FILE, [])


def save_store(store):
    save_json(STORE_FILE, store)


def get_category_items(store, category_id):
    return [item for item in store if item["category_id"] == category_id]


def get_centroid_from_questions(items):
    """
    question_store.json에는 embedding을 저장하지 않으므로,
    비교 시 compare_question으로 다시 임베딩 생성.
    """
    texts = [item["compare_question"] for item in items]
    vectors = [get_embedding(text) for text in texts]
    return np.mean(np.array(vectors), axis=0).tolist()


# 3단계: 카테고리 내부 임베딩 비교
def find_best_group_in_category(question_embedding, category_items):
    if not category_items:
        return None, 0.0

    groups = {}
    for item in category_items:
        groups.setdefault(item["group_id"], []).append(item)

    best_group_id = None
    best_score = -1.0

    for group_id, items in groups.items():
        centroid = get_centroid_from_questions(items)
        score = cosine_similarity(question_embedding, centroid)
        if score > best_score:
            best_score = score
            best_group_id = group_id

    return best_group_id, best_score


def make_new_group_id(store, category_id):
    existing = [
        item["group_id"] for item in store
        if item["category_id"] == category_id and item["group_id"].startswith(f"{category_id}_g")
    ]

    nums = []
    for gid in existing:
        try:
            nums.append(int(gid.split("_g")[-1]))
        except:
            pass

    next_num = max(nums, default=0) + 1
    return f"{category_id}_g{next_num}"


# 전체 파이프라인
def process_question(question: str, categories: list[dict], store: list[dict]):
    classification = classify_question(question, categories)
    category_id = classification["category_id"]
    confidence = classification["confidence"]
    needs_refine = classification["needs_refine"]

    # category_id로 name 찾기
    category_name = next(
        (c["name"] for c in categories if c["id"] == category_id),
        category_id
    )

    compare_question = question

    if needs_refine or len(question) >= LONG_QUESTION_LEN or confidence < 0.65:
        refined = refine_question(question, categories)
        compare_question = refined["refined_question"]
        print(f"[정제됨] {compare_question}")

        classification = classify_question(compare_question, categories)
        category_id = classification["category_id"]
        confidence = classification["confidence"]

        category_name = next(
            (c["name"] for c in categories if c["id"] == category_id),
            category_id
        )

    emb = get_embedding(compare_question)
    category_items = get_category_items(store, category_id)
    best_group_id, best_score = find_best_group_in_category(emb, category_items)

    if best_group_id is None:
        assigned_group = make_new_group_id(store, category_id)
        decision = "새 카테고리 첫 그룹 생성"
    elif best_score >= SIM_THRESHOLD:
        assigned_group = best_group_id
        decision = "기존 그룹에 포함"
    elif best_score >= AMBIGUOUS_LOW:
        assigned_group = make_new_group_id(store, category_id)
        decision = "애매 구간 -> 새 그룹 생성"
    else:
        assigned_group = make_new_group_id(store, category_id)
        decision = "유사도 낮음 -> 새 그룹 생성"

    record = {
        "original_question": question,
        "compare_question": compare_question,
        "category_id": category_id,
        "category_name": category_name,
        "group_id": assigned_group
    }

    store.append(record)
    save_store(store)

    return {
        "original_question": question,
        "compare_question": compare_question,
        "category_id": category_id,
        "category_name": category_name,
        "confidence": confidence,
        "group_id": assigned_group,
        "best_similarity": round(best_score, 4),
        "decision": decision
    }


def print_store_grouped(store, categories):
    category_map = {c["id"]: c["name"] for c in categories}

    grouped = {}
    for item in store:
        key = (item["category_id"], item["group_id"])
        grouped.setdefault(key, []).append(item)

    for (category_id, group_id), items in grouped.items():
        print(f"\n[{category_map.get(category_id, category_id)}] / {group_id}")
        for x in items:
            print(f" - 원문: {x['original_question']}")
            if x["original_question"] != x["compare_question"]:
                print(f"   비교용: {x['compare_question']}")


if __name__ == "__main__":
    categories = initialize_categories()
    store = load_store()

    while True:
        q = input("\n질문 입력 (종료: exit): ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        result = process_question(q, categories, store)

        print("\n[결과]")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print_store_grouped(store, categories)