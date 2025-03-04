from matplotlib import pyplot as plt
from torch_geometric.data import Data
import torch
from pre_solve_dataset.presolve import read_features_names, read_edges, get_node_features
from utils.sampler import NeighborSampler
from models.GraFRank import GraFrank
from models.SAGE import SAGE
from models.Node2Vec import Node2Vec
import torch.nn.functional as F
import math
import random
import networkx as nx

# 定义全局变量
NUM_NODES = 4039
# 选择模型
# MODEL_TYPE = 'GraFrank'
# MODEL_TYPE = 'SAGE'
MODEL_TYPE = 'Node2Vec'


def get_random_index(n, x):
    index = random.sample(range(n), x)
    return index


def get_adjacency_list(edge, user):
    friends = []
    for i in range(0, len(edge)):
        if edge[i][0] == user:
            friends.append(int(edge[i][1]))
        if edge[i][1] == user:
            friends.append(int(edge[i][0]))
    return list(set(friends))


def train(loader):
    model.train()

    total_loss = 0
    it = 0
    for batch_size, n_id, adjs in loader:
        it += 1
        edge_attrs = [data.edge_attr[e_id] for (edge_index, e_id, size) in adjs]
        adjs = [adj.to(device) for adj in adjs]
        edge_attrs = [edge_attr.to(device) for edge_attr in edge_attrs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs, edge_attrs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim = 0)

        # binary skipgram loss can be replaced with margin-based pairwise ranking loss.
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
    model.eval()
    out = model.full_forward(x, edge_index, edge_attr).cpu()
    return out


def calculate_precision_recall_f1(actual, predicted, k):
    """
    计算精确度、召回率和F1值。
    :param actual: 真实的朋友列表。
    :param predicted: 模型预测的朋友列表。
    :param k: 评估的前k个推荐。
    :return: 精确度、召回率和F1值。
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])

    true_positive = len(actual_set & predicted_set)
    precision = true_positive / float(k) if k > 0 else 0
    recall = true_positive / float(len(actual_set)) if len(actual_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1


# 获取特征名称总览
feas = read_features_names('./facebook', '.featnames')
# 读取边
edges = read_edges('./facebook', '.edges')
edges = torch.tensor(edges, dtype=torch.long)
# 处理点特征矩阵
nodes_fea = get_node_features(feas)
# 随机选择95%的数据集为训练集
train1 = torch.ones(NUM_NODES, dtype=bool)
test1 = torch.zeros(NUM_NODES, dtype=bool)
index = get_random_index(NUM_NODES, int(0.05 * NUM_NODES))
for i in range(0, len(index)):
    train1[i] = False
for i in range(0, len(index)):
    test1[index] = True
data = Data(x = nodes_fea, edge_index = edges.t(), train_mask = train1, tesk_mask = test1)
# 边特征维数
n_edge_channels = 5
# 边属性
data.edge_attr = torch.ones([data.edge_index.shape[1], n_edge_channels])

# 创建 NetworkX 图
G = nx.Graph()
# 添加节点到图中
for node in range(NUM_NODES):
    G.add_node(node)
# 添加边到图中
for edge in edges:
    G.add_edge(edge[0], edge[1])


train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256, shuffle=True, num_nodes=data.num_nodes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if MODEL_TYPE == 'GraFrank':
    model = GraFrank(data.num_node_features, hidden_channels=64, edge_channels=n_edge_channels, num_layers=2,
                     input_dim_list=[350, 350,350,356])
    # input dim list assumes that the node features are first
    # partitioned and then concatenated across the K modalities.
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = data.x.to(device)
elif MODEL_TYPE == 'SAGE':
    model = SAGE(data.num_node_features, hidden_channels=64, num_layers=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = data.x.to(device)
elif MODEL_TYPE == 'Node2Vec':
    # 初始化 Node2Vec 实例
    node2vec = Node2Vec(graph=G, dimensions=128, walk_length=10, num_walks=10, p=1, q=1)
    walks = node2vec._generate_walks()
    model = node2vec.fit()


result = torch.tensor((NUM_NODES, NUM_NODES))
for epoch in range(1, 51):
    loss = train(train_loader)
    test()
    if epoch == 50:
        result = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# 当使用 Node2Vec 时，直接从 Word2Vec 模型中提取嵌入向量
if MODEL_TYPE == 'Node2Vec':
    embeddings = {word: model.wv[word] for word in model.wv.key_to_index}
    embeddings_tensor = {word: torch.tensor(embeddings[word], device=device) for word in embeddings}
    # embeddings_tensor 现在包含了节点嵌入向量，可以用于后续任务

ans = torch.zeros((NUM_NODES, NUM_NODES), dtype=float)
# 计算任意两个用户之间的相似度
for i in range(0, NUM_NODES):
    temp1 = result[i]
    temp1_v = temp1 * temp1
    s1 = math.sqrt(temp1_v.sum())
    for j in range(0, NUM_NODES):
        temp2 = result[j]
        r = float(torch.matmul(temp1, temp2))
        temp2_v = temp2 * temp2
        s2 = math.sqrt(temp2_v.sum())
        ans[i][j] = r / (s1 * s2)

sort_ed, indices = torch.sort(ans, dim=1, descending=True)


# 初始化评估指标
total_precision = 0
total_recall = 0
total_f1 = 0

# 遍历每个用户进行评估
for i in range(len(index)):
    user_id = index[i]
    N = 50  # Top-N推荐
    actual_friends = get_adjacency_list(edges, user_id)
    predicted_friends = [int(indices[user_id][j]) for j in range(N)]

    precision, recall, f1 = calculate_precision_recall_f1(actual_friends, predicted_friends, N)
    total_precision += precision
    total_recall += recall
    total_f1 += f1

# 计算平均评估指标
avg_precision = total_precision / len(index)
avg_recall = total_recall / len(index)
avg_f1 = total_f1 / len(index)

print(f"Average Precision@{N}: ")
print(avg_precision)
print(f"Average Recall@{N}: ")
print(avg_recall)
print(f"Average F1@{N}: ")
print(avg_f1)

NDCG = []
hits_list = []
rank = 0
for i in range(0, len(index)):
    DCG = 0
    IDCG = 0
    user_id = i
    N = 50
    # 推荐好友列表
    test_friends = []
    test_friends.clear()
    actual_friends = get_adjacency_list(edges, user_id)
    for j in range(0, N):
        test_friends.append(int(indices[i][j]))
    test_friends = set(test_friends)
    actual_friends = set(actual_friends)
    hits = len(list(test_friends & actual_friends))
    test_friends = list(test_friends)
    actual_friends = list(actual_friends)
    for k in range(0, len(actual_friends)):
        IDCG += 1.0 / math.log2(k + 2)
    for j in range(0, len(test_friends)):
        if test_friends[j] in actual_friends:
            rank += float(1.0 / (j + 1))
            break
    cnt = 0
    for j in range(0, len(test_friends)):
        if test_friends[j] in actual_friends:
            DCG += 1.0 / math.log2(cnt + 2)
            cnt += 1
    # 最大累计增益IDCG不为0
    if IDCG != 0:
        NDCG.append(DCG / IDCG)
    if len(actual_friends) == 0:
        continue
    hitsN_user = hits * 1.0 / len(actual_friends)
    hits_list.append(hitsN_user)

print("hits@50: ")
print(sum(hits_list) / len(hits_list))
print("MRR: ")
print(rank / len(index))
print("NDCG: ")
print(sum(NDCG) / len(NDCG))

torch.save(result, 'result.pt')
torch.save(index, 'index.pt')
