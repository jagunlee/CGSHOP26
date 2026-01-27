import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering

# 1. 거리 행렬 구성 (제공된 데이터 기반)
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

# 대칭 행렬 생성
matrix = np.array(dist_data)
full_matrix = matrix + matrix.T

# 2. MDS를 이용해 2차원 좌표로 변환
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
pos = mds.fit_transform(full_matrix)

# 3. 군집화 수행 (예: 4개의 군집으로 분류)
n_clusters = 4
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
labels = clustering.fit_predict(full_matrix)

# 4. 시각화
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple']

for i in range(n_clusters):
    points = pos[labels == i]
    plt.scatter(points[:, 0], points[:, 1], s=100, label=f'Cluster {i+1}')

# 지점 번호 라벨링
for i, (x, y) in enumerate(pos):
    plt.text(x + 0.1, y + 0.1, str(i), fontsize=12, fontweight='bold')

plt.title("2D Visualization of Clusters (MDS)", fontsize=15)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
