import torch
from torch_geometric.data import Data
import networkx as nx
import torch.nn.functional as F
import math
import random
from pre_solve_dataset.presolve import read_features_names, read_edges, get_node_features
from utils.sampler import NeighborSampler
from models.GraFRank import GraFrank
from models.SAGE import SAGE
from models.Node2Vec import Node2Vec

# 全局变量
NUM_NODES = 4039
N = 50  # Top-N 推荐
MODEL_TYPE = 'GraFrank'  # 'GraFrank', 'SAGE', 'Node2Vec'

# 实用工具函数
def get_random_index(n, x):
    return random.sample(range(n), x)

def get_adjacency_list(edge, user):
    friends = {edge[i][1] for i in range(len(edge)) if edge[i][0] == user}
    friends.update({edge[i][0] for i in range(len(edge)) if edge[i][1] == user})
    return list(friends)

def prepare_masks(num_nodes, train_ratio=0.95):
    """
    准备训练和测试掩码。
    :param num_nodes: 节点总数。
    :param train_ratio: 训练集占总数据集的比例。
    :return: 训练和测试掩码。
    """
    indices = list(range(num_nodes))
    random.shuffle(indices)
    train_size = int(num_nodes * train_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True

    return train_mask, test_mask

# 模型初始化
def initialize_model(model_type, data, device):
    if model_type == 'GraFrank':
        model = GraFrank(data.num_node_features, hidden_channels=64, edge_channels=5, num_layers=2, input_dim_list=[350, 350, 350, 356])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 可以根据需要添加其他模型类型的初始化
    return model, optimizer

# 训练和测试
def train_model(model, loader, data, device, optimizer):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in loader:
        edge_attrs = [data.edge_attr[e_id] for (_, e_id, _) in adjs]
        adjs = [adj.to(device) for adj in adjs]
        edge_attrs = [edge_attr.to(device) for edge_attr in edge_attrs]

        optimizer.zero_grad()
        out = model(data.x[n_id].to(device), adjs, edge_attrs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

    return total_loss / data.num_nodes

def test_model(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model.full_forward(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
    return out.cpu()

# 模型评估
def calculate_precision_recall_f1(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])

    true_positive = len(actual_set & predicted_set)
    precision = true_positive / float(k) if k > 0 else 0
    recall = true_positive / float(len(actual_set)) if len(actual_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1

def evaluate_model(result, edges, index):
    # 初始化评估指标
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    NDCG = []
    hits_list = []
    rank = 0

    # 排序结果，用于获取 Top-N 推荐
    sorted_indices = torch.argsort(result, dim=1, descending=True)

    # 遍历每个用户进行评估
    for user_id in index:
        actual_friends = get_adjacency_list(edges, user_id)
        predicted_friends = [int(sorted_indices[user_id][j]) for j in range(N)]

        # 计算精确度、召回率和 F1 值
        precision, recall, f1 = calculate_precision_recall_f1(actual_friends, predicted_friends, N)
        total_precision += precision
        total_recall += recall
        total_f1 += f1

        # 计算 NDCG 和 MRR
        DCG = 0
        IDCG = 0
        test_friends = set(predicted_friends)
        actual_friends_set = set(actual_friends)
        hits = len(test_friends & actual_friends_set)
        for j, friend in enumerate(predicted_friends):
            if friend in actual_friends_set:
                DCG += 1.0 / math.log2(j + 2)
                if rank == 0:
                    rank += 1.0 / (j + 1)
        for k, friend in enumerate(actual_friends):
            IDCG += 1.0 / math.log2(k + 2)
        if IDCG != 0:
            NDCG.append(DCG / IDCG)
        hitsN_user = hits / len(actual_friends_set) if actual_friends_set else 0
        hits_list.append(hitsN_user)

    # 计算平均评估指标
    avg_precision = total_precision / len(index)
    avg_recall = total_recall / len(index)
    avg_f1 = total_f1 / len(index)
    avg_NDCG = sum(NDCG) / len(NDCG) if NDCG else 0
    MRR = rank / len(index)
    avg_hits = sum(hits_list) / len(hits_list) if hits_list else 0

    return avg_precision, avg_recall, avg_f1, avg_NDCG, MRR, avg_hits

# 主执行逻辑
def main():
    # 数据预处理
    feas = read_features_names('./facebook', '.featnames')
    edges = read_edges('./facebook', '.edges')
    nodes_fea = get_node_features(feas)

    # 假设每条边有1个特征
    num_edges = len(edges)
    num_edge_features = 4039  # 根据实际情况调整这个值
    edge_attrs = torch.ones((num_edges, num_edge_features), dtype=torch.float32)

    # 数据准备
    data = Data(x=nodes_fea, edge_index=torch.tensor(edges, dtype=torch.long).t())
    train_mask, test_mask = prepare_masks(NUM_NODES)

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer = initialize_model(MODEL_TYPE, data, device)

    # 模型训练和测试
    train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256, shuffle=True, num_nodes=data.num_nodes)
    for epoch in range(1, 51):
        loss = train_model(model, train_loader, data, device, optimizer)
        result = test_model(model, data, device)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # 模型评估
    evaluate_model(result, edges, NUM_NODES)

    # 保存结果
    torch.save(result, 'result.pt')

if __name__ == "__main__":
    main()
