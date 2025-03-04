import os
import torch

def read_features_names(path, suffix):
    """
    批量读取特征名称并去重。
    :param path: 特征文件所在目录。
    :param suffix: 文件后缀。
    :return: 唯一的特征名称列表。
    """
    files = os.listdir(path)
    unique_features = set()
    for file in files:
        if os.path.splitext(file)[1] == suffix:
            with open(os.path.join(path, file), 'r') as f:
                for row in f:
                    feature_name = ' '.join(row.split()[1:4])
                    unique_features.add(feature_name)
    return list(unique_features)

def read_edges(path, suffix):
    """
    读取边。
    :param path: 边文件所在目录。
    :param suffix: 文件后缀。
    :return: 唯一边的集合。
    """
    files = os.listdir(path)
    edges = set()
    for file in files:
        if os.path.splitext(file)[1] == suffix:
            with open(os.path.join(path, file), 'r') as f:
                for row in f:
                    nodes = tuple(map(int, row.split()[:2]))
                    edges.add(nodes)
                    edges.add(nodes[::-1])  # 双向边
    return list(edges)

def get_node_features(feas, num_nodes=4039, num_features=1406):
    """
    获取节点特征。
    :param feas: 特征名称列表。
    :param num_nodes: 节点数量。
    :param num_features: 特征维数。
    :return: 节点特征张量。
    """
    nodes_fea = torch.zeros([num_nodes, num_features], dtype=torch.float32)
    circles_list = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    for ego in circles_list:
        featnames_path = os.path.join('./facebook', f'{ego}.featnames')
        with open(featnames_path, 'r') as f1:
            fea_list = [feas.index(' '.join(row.split()[1:4])) for row in f1]

        egofeat_path = os.path.join('./facebook', f'{ego}.egofeat')
        with open(egofeat_path, 'r') as f2:
            for row in f2:
                nodes_fea[ego, fea_list] = torch.tensor(list(map(int, row.split())), dtype=torch.float32)

        feat_path = os.path.join('./facebook', f'{ego}.feat')
        with open(feat_path, 'r') as f3:
            for row in f3:
                data = list(map(int, row.split()))
                x = data[0]
                nodes_fea[x, fea_list] = torch.tensor(data[1:], dtype=torch.float32)
    return nodes_fea
