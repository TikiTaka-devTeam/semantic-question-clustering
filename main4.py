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
GRAPH_FILE = "graph_store.json"

EMBEDDING_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"

SIM_THRESHOLD = 0.80
MULTI_CAT_THRESHOLD = 0.65
LONG_QUESTION_LEN = 80


# ── 공통 유틸 ──────────────────────────────────────────────

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
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    text = resp.choices[0].message.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    return json.loads(text)


# ── 1단계: PDF 기반 카테고리 생성 ──────────────────────────

def generate_categories_from_material(material_text: str) -> list[dict]:
    prompt = f"""
너는 강의자료 분석 도우미다.
아래 강의자료를 보고, 학생 질문 분류에 적합한 카테고리를 생성하라.

조건:
- 카테고리는 5~10개
- 너무 겹치지 않게 만들 것
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


# ── 2단계: 질문 카테고리 분류 ──────────────────────────────

def classify_question(question: str, categories: list[dict]) -> dict:
    category_text = "\n".join([
        f"- {c['id']} | {c['name']} | {c['description']} | keywords={', '.join(c['keywords'])}"
        for c in categories
    ])

    prompt = f"""
너는 학생 질문을 강의 카테고리에 분류하는 분류기다.

아래 카테고리 목록에서 질문에 해당하는 카테고리를 고르되,
질문이 여러 카테고리에 걸쳐있을 경우 복수로 선택할 수 있다.

각 카테고리마다 confidence를 0~1 사이로 추정하라.
confidence가 {MULTI_CAT_THRESHOLD} 이상인 카테고리를 모두 포함하라.

질문이 장황하거나 의도가 불명확하면 needs_refine을 true로 하라.

반드시 아래 JSON 형식으로만 답하라.
{{
  "primary_category_id": "cat1",
  "categories": [
    {{"category_id": "cat1", "confidence": 0.91}},
    {{"category_id": "cat2", "confidence": 0.72}}
  ],
  "needs_refine": false,
  "reason": "짧게"
}}

카테고리:
{category_text}

질문:
{question}
"""
    return call_json_response(prompt)


# ── 3단계: 질문 정제 ─────────────────────────────────────

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


# ── 그래프 저장소 관리 ────────────────────────────────────

def load_graph() -> dict:
    default = {
        "nodes": {
            "questions": [],
            "categories": []
        },
        "edges": {
            "belongs_to": [],
            "similar_to": []
        }
    }
    return load_json(GRAPH_FILE, default)


def save_graph(graph: dict):
    save_json(GRAPH_FILE, graph)


def get_all_questions(graph: dict) -> list[dict]:
    return graph["nodes"]["questions"]


def find_question_by_id(graph: dict, qid: str) -> dict | None:
    for q in graph["nodes"]["questions"]:
        if q["id"] == qid:
            return q
    return None


def add_question_node(
    graph: dict,
    original: str,
    compare: str,
    embedding: list,
    page: int
) -> str:
    qid = "q_" + str(uuid.uuid4())[:8]

    graph["nodes"]["questions"].append({
        "id": qid,
        "page": page,
        "original": original,
        "compare": compare,
        "embedding": embedding
    })

    return qid


def add_belongs_to_edge(graph: dict, question_id: str, category_id: str, confidence: float):
    for edge in graph["edges"]["belongs_to"]:
        if edge["from"] == question_id and edge["to"] == category_id:
            return

    graph["edges"]["belongs_to"].append({
        "from": question_id,
        "to": category_id,
        "confidence": round(confidence, 4)
    })


def add_similar_to_edge(graph: dict, id_a: str, id_b: str, score: float):
    for edge in graph["edges"]["similar_to"]:
        if {edge["from"], edge["to"]} == {id_a, id_b}:
            return

    graph["edges"]["similar_to"].append({
        "from": id_a,
        "to": id_b,
        "score": round(score, 4)
    })


# ── 유사 질문 엣지 생성 ───────────────────────────────────

def compute_similarity_edges(graph: dict, new_id: str, new_embedding: list):
    for q in graph["nodes"]["questions"]:
        if q["id"] == new_id:
            continue

        score = cosine_similarity(new_embedding, q["embedding"])

        if score >= SIM_THRESHOLD:
            add_similar_to_edge(graph, new_id, q["id"], score)
            print(f"  [similar_to] {new_id} ↔ {q['id']} (score={score:.4f})")


# ── 전체 파이프라인 ──────────────────────────────────────

def process_question(
    question: str,
    page: int,
    categories: list[dict],
    graph: dict
) -> dict:
    classification = classify_question(question, categories)
    needs_refine = classification["needs_refine"]
    compare_question = question

    first_confidence = classification.get("categories", [{}])[0].get("confidence", 1)

    if needs_refine or len(question) >= LONG_QUESTION_LEN or first_confidence < 0.65:
        refined = refine_question(question, categories)
        compare_question = refined["refined_question"]

        print(f"  [정제됨] {compare_question}")

        classification = classify_question(compare_question, categories)

    embedding = get_embedding(compare_question)

    qid = add_question_node(
        graph=graph,
        original=question,
        compare=compare_question,
        embedding=embedding,
        page=page
    )

    assigned_categories = []

    for cat_info in classification.get("categories", []):
        cat_id = cat_info["category_id"]
        confidence = cat_info["confidence"]

        if confidence >= MULTI_CAT_THRESHOLD:
            add_belongs_to_edge(graph, qid, cat_id, confidence)
            assigned_categories.append({
                "category_id": cat_id,
                "confidence": confidence
            })

    compute_similarity_edges(graph, qid, embedding)

    save_graph(graph)

    return {
        "question_id": qid,
        "page": page,
        "original": question,
        "compare": compare_question,
        "belongs_to": assigned_categories,
        "similar_to_count": sum(
            1 for e in graph["edges"]["similar_to"]
            if e["from"] == qid or e["to"] == qid
        )
    }


# ── 페이지별 질문 출력 ────────────────────────────────────

def print_questions_by_page(graph: dict):
    questions = graph["nodes"]["questions"]

    page_map = {}

    for q in questions:
        page = q.get("page", "페이지 없음")
        page_map.setdefault(page, []).append(q)

    print("\n[페이지별 질문]")

    for page in sorted(page_map.keys(), key=lambda x: int(x) if isinstance(x, int) or str(x).isdigit() else 9999):
        print(f"\n  [Page {page}]")

        for q in page_map[page]:
            print(f"    - ({q['id']}) {q['original']}")

            if q["original"] != q["compare"]:
                print(f"      비교용: {q['compare']}")


def print_questions_for_page(graph: dict, page: int):
    questions = graph["nodes"]["questions"]
    results = [q for q in questions if q.get("page") == page]

    print(f"\n[Page {page} 질문]")

    if not results:
        print("  해당 페이지 질문이 없습니다.")
        return

    for q in results:
        print(f"  - ({q['id']}) {q['original']}")

        if q["original"] != q["compare"]:
            print(f"    비교용: {q['compare']}")


# ── 그래프 기반 요약 출력 ─────────────────────────────────

def print_graph_summary(graph: dict, categories: list[dict]):
    cat_map = {c["id"]: c["name"] for c in categories}
    questions = graph["nodes"]["questions"]
    belongs_to = graph["edges"]["belongs_to"]
    similar_to = graph["edges"]["similar_to"]

    print("\n" + "=" * 60)
    print(f"  총 질문 수: {len(questions)}")
    print(f"  belongs_to 엣지 수: {len(belongs_to)}")
    print(f"  similar_to 엣지 수: {len(similar_to)}")
    print("=" * 60)

    cat_questions = {}

    for edge in belongs_to:
        qid = edge["from"]
        cat_id = edge["to"]
        cat_questions.setdefault(cat_id, []).append((qid, edge["confidence"]))

    print("\n[카테고리별 질문]")

    for cat_id, items in cat_questions.items():
        print(f"\n  [{cat_map.get(cat_id, cat_id)}]")

        for qid, conf in sorted(items, key=lambda x: -x[1]):
            q = find_question_by_id(graph, qid)

            if q:
                multi = sum(1 for e in belongs_to if e["from"] == qid)
                multi_tag = " ★복수카테고리" if multi > 1 else ""
                page_info = f"Page {q.get('page', '?')}"

                print(f"    - [{conf:.2f}] {page_info}{multi_tag} | {q['original']}")

    print("\n[유사도 높은 질문 쌍 similar_to]")

    sorted_edges = sorted(similar_to, key=lambda x: -x["score"])

    if not sorted_edges:
        print("  아직 유사 질문 연결이 없습니다.")
        return

    for edge in sorted_edges[:10]:
        qa = find_question_by_id(graph, edge["from"])
        qb = find_question_by_id(graph, edge["to"])

        if qa and qb:
            print(f"\n  {edge['score']:.4f}")
            print(f"    - Page {qa.get('page', '?')} | {qa['original']}")
            print(f"    - Page {qb.get('page', '?')} | {qb['original']}")


# ── 유사 질문 묶음 출력 ───────────────────────────────────

def get_connected_components(graph: dict):
    questions = graph["nodes"]["questions"]
    similar_to = graph["edges"]["similar_to"]

    node_ids = [q["id"] for q in questions]
    adjacency = {qid: [] for qid in node_ids}

    for edge in similar_to:
        adjacency[edge["from"]].append(edge["to"])
        adjacency[edge["to"]].append(edge["from"])

    visited = set()
    components = []

    for qid in node_ids:
        if qid in visited:
            continue

        stack = [qid]
        component = []
        visited.add(qid)

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
    components = get_connected_components(graph)

    print("\n[비슷한 질문 묶음]")

    group_no = 1

    for component in components:
        if len(component) < 2:
            continue

        print(f"\n  [유사 질문 그룹 {group_no}]")

        for qid in component:
            q = find_question_by_id(graph, qid)

            if q:
                print(f"    - Page {q.get('page', '?')} / {q['id']}: {q['compare']}")

        group_no += 1

    if group_no == 1:
        print("  아직 유사 질문 그룹이 없습니다.")


# ── 메인 ─────────────────────────────────────────────────

if __name__ == "__main__":
    categories = initialize_categories()
    graph = load_graph()

    graph["nodes"]["categories"] = categories
    save_graph(graph)

    while True:
        print("\n==============================")
        print("1. 질문 입력")
        print("2. 전체 요약 보기")
        print("3. 페이지별 질문 보기")
        print("4. 특정 페이지 질문 보기")
        print("5. 비슷한 질문 묶음 보기")
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

            q = input("질문 입력: ").strip()

            if not q:
                print("질문이 비어 있습니다.")
                continue

            result = process_question(q, page, categories, graph)

            print("\n[결과]")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        elif menu == "2":
            print_graph_summary(graph, categories)

        elif menu == "3":
            print_questions_by_page(graph)

        elif menu == "4":
            page_input = input("조회할 페이지 번호 입력: ").strip()

            if not page_input.isdigit():
                print("페이지 번호는 숫자로 입력해야 합니다.")
                continue

            print_questions_for_page(graph, int(page_input))

        elif menu == "5":
            print_similar_question_groups(graph)

        else:
            print("잘못된 메뉴입니다.")

    print_graph_summary(graph, categories)