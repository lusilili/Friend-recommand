import os
import pandas as pd
import torch
from train import NUM_NODES, index  # 使用训练集中定义的num_nodes

# 基于训练结果，对于节点进行分数评价用于实现好友推荐
result = torch.load('result.pt')


# 基于训练得到的结果，计算所有用户之间的相似度
def compute_similarity(result):
    similarity = torch.zeros((NUM_NODES, NUM_NODES), dtype=torch.float)
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            dot_product = torch.dot(result[i], result[j])
            norm_i = result[i].norm(p=2)
            norm_j = result[j].norm(p=2)
            similarity[i][j] = dot_product / (norm_i * norm_j)
    return similarity


# 根据相似度矩阵推荐好友
def recommend_friends(similarity_matrix, user_id, top_k):
    _, top_indices = torch.topk(similarity_matrix[user_id], top_k + 1)
    # 排除自身，只推荐其他用户
    top_indices = top_indices[1:]
    return top_indices.tolist()


# 对每个用户进行好友推荐
def make_recommendations(test_indices, similarity_matrix, top_k):
    recommendations = {}
    for user_id in test_indices:
        recommended_friends = recommend_friends(similarity_matrix, user_id, top_k)
        recommendations[user_id] = recommended_friends
    return recommendations


# 计算推荐结果的相似度矩阵
similarity_matrix = compute_similarity(result)

# 为测试集中的用户推荐好友
top_k = 50  # 推荐列表中的好友数量
recommendations = make_recommendations(index, similarity_matrix, top_k)

# 打印推荐结果
# 将推荐结果存入Excel文件
df = pd.DataFrame.from_dict(recommendations, orient='index', columns=[f"Friend_{i+1}" for i in range(top_k)])
df.index.name = 'User_ID'
# 判断文件是否存在
filename = 'recommendations.xlsx'
if os.path.exists(filename):
    # 文件存在，则在文件名后添加数字存储
    filename = filename + '_' + str(len(os.listdir('.'))) + '.xlsx'

# 保存文件
df.to_excel(filename)