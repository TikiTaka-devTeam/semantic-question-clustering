import os
import json
import uuid
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_FILE = "2-3)_2장._OSI_모델과_TCP-IP_프로토콜_(31_pages).pdf"
CATEGORY_FILE = "categories.json"
GRAPH_FILE = "question_graph.json"
EMBEDDING_STORE_FILE = "question_embeddings.json"

EMBEDDING_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-5.4-mini"

EDGE_THRESHOLD = 0.80
LONG_QUESTION_LEN = 80


# ---------------- 공통 유틸 ----------------

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

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

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


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)

    return "\n".join(texts).strip()


# ---------------- 카테고리 생성 ----------------

def generate_categories_from_material(material_text: str) -> list[dict]:
    prompt = f"""
너는 강의자료 분석 도우미다.
아래 강의자료를 보고, 학생 질문 분류에 적합한 카테고리를 생성하라.

조건:
- 카테고리는 5~10개
- 너무 겹치지 않게 만들 것
- 각 카테고리마다 id, name, description, keywords 포함

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


# ---------------- 질문 분류 / 정제 ----------------

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


def refine_question(question: str, categories: list[dict]) -> dict:
    category_names = ", ".join([c["name"] for c in categories])

    prompt = f"""
너는 학생 질문 정제 도우미다.
아래 장황하거나 애매한 질문을 핵심 의미를 유지하면서 짧고 비교하기 쉬운 질문 한 문장으로 바꿔라.

조건:
- 인사말, 군더더기 제거
- 질문 의도가 여러 개면 가장 핵심 질문 하나만 남김
- 가능한 한 아래 카테고리 중 하나에 잘 대응되도록 정리
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


# ---------------- 그래프 / 임베딩 저장소 ----------------

def load_graph():
    return load_json(GRAPH_FILE, {
        "nodes": [],
        "edges": []
    })


def save_graph(graph):
    save_json(GRAPH_FILE, graph)


def load_embedding_store():
    return load_json(EMBEDDING_STORE_FILE, {})


def save_embedding_store(embedding_store):
    save_json(EMBEDDING_STORE_FILE, embedding_store)


def make_question_id():
    return "q_" + uuid.uuid4().hex[:8]


def edge_exists(edges, source, target):
    for edge in edges:
        if edge["source"] == source and edge["target"] == target:
            return True
        if edge["source"] == target and edge["target"] == source:
            return True
    return False


# ---------------- 그래프 기반 질문 처리 ----------------

def process_question_graph(
    question: str,
    page: int,
    categories: list[dict],
    graph: dict,
    embedding_store: dict
):
    classification = classify_question(question, categories)

    category_id = classification["category_id"]
    confidence = classification["confidence"]
    needs_refine = classification["needs_refine"]

    compare_question = question

    if needs_refine or len(question) >= LONG_QUESTION_LEN or confidence < 0.65:
        refined = refine_question(question, categories)
        compare_question = refined["refined_question"]

        print(f"[정제됨] {compare_question}")

        classification = classify_question(compare_question, categories)
        category_id = classification["category_id"]
        confidence = classification["confidence"]

    new_embedding = get_embedding(compare_question)

    new_node = {
        "id": make_question_id(),
        "page": page,
        "original_question": question,
        "compare_question": compare_question,
        "category_id": category_id,
        "confidence": confidence
    }

    connected_edges = []

    for node in graph["nodes"]:
        old_embedding = embedding_store.get(node["id"])

        if old_embedding is None:
            continue

        score = cosine_similarity(new_embedding, old_embedding)

        if score >= EDGE_THRESHOLD:
            edge = {
                "source": new_node["id"],
                "target": node["id"],
                "similarity": round(score, 4)
            }

            if not edge_exists(graph["edges"], edge["source"], edge["target"]):
                graph["edges"].append(edge)
                connected_edges.append(edge)

    embedding_store[new_node["id"]] = new_embedding
    graph["nodes"].append(new_node)

    save_graph(graph)
    save_embedding_store(embedding_store)

    return {
        "question_id": new_node["id"],
        "page": page,
        "original_question": question,
        "compare_question": compare_question,
        "category_id": category_id,
        "confidence": round(confidence, 4),
        "connected_count": len(connected_edges),
        "connected_edges": connected_edges
    }


# ---------------- 페이지별 질문 보기 ----------------

def print_questions_by_page(graph: dict):
    page_map = {}

    for node in graph["nodes"]:
        page_map.setdefault(node["page"], []).append(node)

    print("\n========== 페이지별 질문 ==========")

    for page in sorted(page_map.keys()):
        print(f"\n[Page {page}]")

        for node in page_map[page]:
            print(f"- ({node['id']}) {node['original_question']}")

            if node["original_question"] != node["compare_question"]:
                print(f"  비교용: {node['compare_question']}")


def print_page_questions(graph: dict, page: int):
    results = [node for node in graph["nodes"] if node["page"] == page]

    print(f"\n========== Page {page} 질문 ==========")

    if not results:
        print("해당 페이지 질문이 없습니다.")
        return

    for node in results:
        print(f"- ({node['id']}) {node['original_question']}")

        if node["original_question"] != node["compare_question"]:
            print(f"  비교용: {node['compare_question']}")


# ---------------- 유사 질문 묶음 보기 ----------------

def get_connected_components(graph: dict):
    node_ids = [node["id"] for node in graph["nodes"]]
    visited = set()

    adjacency = {node_id: [] for node_id in node_ids}

    for edge in graph["edges"]:
        adjacency[edge["source"]].append(edge["target"])
        adjacency[edge["target"]].append(edge["source"])

    components = []

    for node_id in node_ids:
        if node_id in visited:
            continue

        stack = [node_id]
        component = []
        visited.add(node_id)

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def print_similar_question_groups(graph: dict):
    node_map = {node["id"]: node for node in graph["nodes"]}
    components = get_connected_components(graph)

    print("\n========== 비슷한 질문 묶음 ==========")

    group_num = 1

    for component in components:
        if len(component) < 2:
            continue

        print(f"\n[유사 질문 그룹 {group_num}]")

        for node_id in component:
            node = node_map[node_id]
            print(f"- Page {node['page']} / {node['id']}: {node['compare_question']}")

        group_num += 1

    if group_num == 1:
        print("\n아직 유사 질문 그룹이 없습니다.")


# ---------------- 특정 질문과 비슷한 질문 보기 ----------------

def print_similar_questions_for_node(graph: dict, question_id: str):
    node_map = {node["id"]: node for node in graph["nodes"]}

    if question_id not in node_map:
        print("해당 question_id가 없습니다.")
        return

    print(f"\n========== {question_id}와 비슷한 질문 ==========")
    print(f"기준 질문: {node_map[question_id]['compare_question']}")

    found = False

    for edge in graph["edges"]:
        if edge["source"] == question_id:
            other_id = edge["target"]
        elif edge["target"] == question_id:
            other_id = edge["source"]
        else:
            continue

        other = node_map[other_id]
        print(
            f"- similarity={edge['similarity']} / "
            f"Page {other['page']} / {other['id']}: {other['compare_question']}"
        )
        found = True

    if not found:
        print("연결된 유사 질문이 없습니다.")


# ---------------- 실행 ----------------

if __name__ == "__main__":
    categories = initialize_categories()
    graph = load_graph()
    embedding_store = load_embedding_store()

    while True:
        print("\n==============================")
        print("1. 질문 입력")
        print("2. 페이지별 질문 보기")
        print("3. 비슷한 질문 묶음 보기")
        print("4. 특정 페이지 질문 보기")
        print("5. 특정 질문과 비슷한 질문 보기")
        print("0. 종료")
        print("==============================")

        menu = input("메뉴 선택: ").strip()

        if menu == "0":
            break

        elif menu == "1":
            page_input = input("페이지 번호 입력: ").strip()

            if not page_input.isdigit():
                print("페이지 번호는 숫자로 입력해야 합니다.")
                continue

            page = int(page_input)

            question = input("질문 입력: ").strip()

            if not question:
                print("질문이 비어 있습니다.")
                continue

            result = process_question_graph(
                question,
                page,
                categories,
                graph,
                embedding_store
            )

            print("\n[결과]")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif menu == "2":
            print_questions_by_page(graph)

        elif menu == "3":
            print_similar_question_groups(graph)

        elif menu == "4":
            page_input = input("조회할 페이지 번호 입력: ").strip()

            if not page_input.isdigit():
                print("페이지 번호는 숫자로 입력해야 합니다.")
                continue

            print_page_questions(graph, int(page_input))

        elif menu == "5":
            question_id = input("question_id 입력: ").strip()
            print_similar_questions_for_node(graph, question_id)

        else:
            print("잘못된 메뉴입니다.")