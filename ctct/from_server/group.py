import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# 1. 거리 행렬 데이터
dist_data = [
    [0, 5, 5, 5, 11, 4, 8, 9, 8, 5, 5, 8, 6, 8, 5, 5, 5, 7, 6, 7],
    [0, 0, 6, 5, 9, 6, 7, 6, 5, 6, 5, 6, 6, 6, 5, 4, 5, 6, 5, 6],
    [0, 0, 0, 6, 8, 6, 6, 8, 6, 4, 6, 6, 5, 7, 5, 5, 5, 5, 4, 5],
    [0, 0, 0, 0, 8, 7, 6, 7, 8, 6, 6, 6, 5, 7, 6, 5, 5, 6, 6, 5],
    [0, 0, 0, 0, 0, 7, 5, 8, 7, 7, 8, 7, 8, 5, 6, 7, 7, 6, 6, 5],
    [0, 0, 0, 0, 0, 0, 6, 8, 6, 4, 5, 6, 6, 6, 6, 5, 6, 5, 5, 6],
    [0, 0, 0, 0, 0, 0, 0, 6, 6, 4, 5, 3, 5, 5, 6, 4, 6, 5, 4, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 6, 5, 7, 5, 6, 7, 9, 9, 7, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 6, 7, 7, 5, 7, 5, 7, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 4, 5, 6, 5, 5, 4, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 6, 5, 4, 6, 6, 5, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 5, 5, 8, 5, 5, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 4, 5, 5, 4, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 7, 6, 4, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 5, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# 대칭 행렬 변환
matrix = np.array(dist_data)
full_matrix = matrix + matrix.T

# 2. 사용자 정의 그룹 설정 (pfd 기준)
pfd_groups = {
    'pfd 003': [2, 5, 9, 11, 14, 15, 17, 18, 19],
    'pfd 004': [1, 3, 6, 8, 10, 12, 13, 16],
    'pfd 005': [0, 4, 7]
}

# 각 포인트가 어떤 그룹인지 라벨 생성
labels = np.zeros(20, dtype=int)
group_names = list(pfd_groups.keys())
for i, name in enumerate(group_names):
    for point in pfd_groups[name]:
        labels[point] = i

# 3. MDS 차원 축소 (2D 좌표 생성)
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
pos = mds.fit_transform(full_matrix)

# 4. 시각화
plt.figure(figsize=(11, 8))
colors = ['#FF9999', '#66B2FF', '#99FF99'] # 부드러운 빨강, 파랑, 초록

for i, name in enumerate(group_names):
    points = pos[labels == i]
    plt.scatter(points[:, 0], points[:, 1], s=250, c=colors[i], label=name, edgecolors='black', alpha=0.8)

# 포인트 번호 표시
for i, (x, y) in enumerate(pos):
    plt.text(x, y, str(i), fontsize=10, ha='center', va='center', fontweight='bold')

plt.title("MDS Visualization by PFD Groups", fontsize=16, pad=20)
plt.xlabel("Coordinate X")
plt.ylabel("Coordinate Y")
plt.legend(title="Groups", loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

# 그룹 간의 물리적 경계를 시각적으로 이해하기 돕기 위한 보조선 (선택사항)
plt.tight_layout()
plt.show()
