# ---------运行结果代码------------
import os
import numpy as np

import torch
from cnn_model import RNAInception

def process_bpseq_file(file_path):  # 非全连接的rna边信息
    sequences = []
    list1 = []
    list2 = []

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            index, base, pair_index = line.strip().split()
            index = int(index)-1
            pair_index = int(pair_index)-1
            sequences.append(base)
            if int(index) != -1 and int(pair_index) != -1:
                list1.append(int(index))
                list2.append(int(pair_index))
        edge_matrix = np.array([list1, list2])

    return sequences, edge_matrix


def get_input(input_path, cor_path, true_path=None):
    max_len = 734

    struc_matrix, seq = None, None

    try:
        struc_path = input_path
        if os.path.exists(struc_path):
            seq, edge = process_bpseq_file(struc_path)
            edge_index = torch.LongTensor(edge)

            adj_matrix = torch.zeros((len(seq), len(seq)))
            adj_matrix[edge_index[0], edge_index[1]] = 1
            adj_matrix = adj_matrix.double()
        else:
            print("Error")

        embedding_path = cor_path
        if os.path.exists(embedding_path):
            emb = np.load(embedding_path)
            emb = torch.from_numpy(emb)
            emb = emb.double()
            flatten_x = torch.zeros((len(seq), 645))  # 补齐data.x
            flatten_x[:emb.shape[0], :emb.shape[1]] = emb
            flatten_x = flatten_x.double()
        else:
            print("Error")

    except Exception as e:
        print(f"error:{e}")

    length = len(seq)
    print(f"RNA序列的长度为{length}\n")

    struc_matrix = adj_matrix[:length, :length]
    struc_matrix = struc_matrix.unsqueeze(0)
    seq_emb = flatten_x[:length, :]
    seq_emb = seq_emb.unsqueeze(0)

    pre_matrix = struc_matrix[0, :, :]
    pre_matrix = pre_matrix.numpy()

    if true_path == None:
        pass
    else:
        true_seq,true_edge = process_bpseq_file(true_path)
        true_matrix = torch.zeros((length, length))
        true_matrix[true_edge[0], true_edge[1]] = 1
        true_matrix = true_matrix.numpy()

    return struc_matrix, seq_emb, pre_matrix, true_matrix

def run():
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    true_path = r"" # 可省略
    input_path_npy = r"/home/Shihx/RNA-FM/redevelop/TestA/representations/370.npy"  # 输入结构对应的大语言模型embedding
    input_path = r"" # 此处输入预测结构

    model = RNAInception(out_channels=512)
    state_dict = torch.load("checkpoint/4_layer.pth",
                            map_location=torch.device("cpu"))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # Remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)

    input_struc, input_seq, pre_matrix, true_struc = get_input(input_path, input_path_npy, true_path)

    true_matrix = torch.from_numpy(true_struc).unsqueeze(0)
    input_seq = input_seq.float().to(device)
    input_struc = true_matrix.float().to(device) # 真实结构
    # input_struc = input_struc.float().to(device) # 预测结构

    outputs = model(input_struc, input_seq)
    predict = torch.argmax(outputs, 1)
    result = "正样本" if predict.item() == 1 else "负样本"
    print(f"该预测结果为: {result}")

if __name__ == '__main__':
    run()