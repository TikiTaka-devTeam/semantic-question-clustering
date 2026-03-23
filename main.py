from openai import OpenAI
import numpy as np
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 질문 불러오기
with open("questions.txt", "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

# 임베딩 한 번에 생성
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=questions
)

embeddings = [item.embedding for item in response.data]

# 코사인 유사도
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 그룹의 중심 벡터 계산
def get_centroid(group_indices):
    vecs = [embeddings[i] for i in group_indices]
    return np.mean(vecs, axis=0)

# 그룹핑
threshold = 0.8
groups = []

for idx, emb in enumerate(embeddings):
    placed = False

    for group in groups:
        centroid = get_centroid(group)
        if cosine_similarity(emb, centroid) > threshold:
            group.append(idx)
            placed = True
            break

    if not placed:
        groups.append([idx])

# 결과 출력
for i, g in enumerate(groups):
    print(f"\nGroup {i+1}:")
    for idx in g:
        print(f"  - {questions[idx]}")

print(f"\n총 그룹 수: {len(groups)}")