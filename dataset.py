import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import norm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from rna_tools.SecondaryStructure import parse_vienna_to_pairs
from ViennaRNA import RNA
import random

def encode_onehot(input):
    list_RNA = ['A', 'U', 'G', 'C','0']
    classes = set(list_RNA)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, input)),
                             dtype=np.int32)
    return labels_onehot

class RNASeqDataset(Dataset):
    def __init__(self, bpseq_pos_dir, bpseq_neg_dir, rna_fm_dir):
        self.bpseq_pos_dir = bpseq_pos_dir
        self.bpseq_neg_dir = bpseq_neg_dir
        self.rna_fm_dir = rna_fm_dir

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        # 加载正样本
        for file_name in os.listdir(self.bpseq_pos_dir):
            if file_name.endswith('.bpseq'):
                bpseq_path = os.path.join(self.bpseq_pos_dir, file_name)
                npy_path = os.path.join(self.rna_fm_dir, file_name.replace('.bpseq', '.npy'))
                if os.path.exists(npy_path):
                    self.samples.append((bpseq_path, npy_path, 1))  # 1 是正样本标签

        # 加载负样本
        for file_name in os.listdir(self.bpseq_neg_dir):
            if file_name.endswith('.bpseq'):
                bpseq_path = os.path.join(self.bpseq_neg_dir, file_name)
                npy_path = os.path.join(self.rna_fm_dir, file_name.replace('.bpseq', '.npy'))
                if os.path.exists(npy_path):
                    self.samples.append((bpseq_path, npy_path, 0))  # 0 是负样本标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_len = 734
        bpseq_path, npy_path, label = self.samples[idx]

        # 读取bpseq数据（这里可以根据需要进行解析）
        seq, edge = self._parse_bpseq(bpseq_path)
        edge_index = torch.LongTensor(edge)

        adj_matrix = torch.zeros((max_len, max_len))
        adj_matrix[edge_index[0], edge_index[1]] = 1
        adj_matrix = adj_matrix.double()


        # 读取RNA-FM数据
        valid_elements = {'A', 'U', 'G', 'C', '0'}
        replaced_data = np.where(np.isin(seq, list(valid_elements)), seq, '0')
        features = encode_onehot(replaced_data)
        features = torch.from_numpy(features)

        rna_fm_data = np.load(npy_path)
        rna_fm_data = rna_fm_data.astype(float)
        rna_fm_data = torch.from_numpy(rna_fm_data)
        x_features = [features, rna_fm_data]
        x_features = torch.cat(x_features, dim=1)
        x_features = x_features.double()

        flatten_x = torch.zeros((max_len, 645))  # 补齐data.x
        flatten_x[:x_features.shape[0], :x_features.shape[1]] = x_features
        flatten_x = flatten_x.double()

        return adj_matrix, flatten_x, label

    def _parse_bpseq(self, bpseq_path):
        sequences = []
        list1 = []
        list2 = []

        with open(bpseq_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                index, base, pair_index = line.strip().split()
                index = int(index) - 1
                pair_index = int(pair_index) - 1
                sequences.append(base)
                if int(index) != -1 and int(pair_index) != -1:
                    list1.append(int(index))
                    list2.append(int(pair_index))
            edge_matrix = np.array([list1, list2])

        return sequences, edge_matrix


class RNA_family_Dataset(Dataset):
    def __init__(self, family_dir, rna_fm_dir,tmRNA_dir, tmRNA_fm_dir):
        self.family_dir = family_dir
        self.rna_fm_dir = rna_fm_dir
        self.tmRNA_fm_dir = tmRNA_fm_dir
        self.tmRNA_dir = tmRNA_dir

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        if os.path.exists(self.tmRNA_dir):
            # 遍历 base_dir 目录下的所有文件夹
            for folder in os.listdir(self.tmRNA_dir):
                folder_path = os.path.join(self.tmRNA_dir, folder)

                if not os.path.isdir(folder_path):
                    continue  # 跳过非文件夹

                # 只处理 xxx（正样本）和 xxx_best（负样本），忽略 xxx_worst
                if folder.endswith("_worst"):  # _worst
                    continue

                label = 1 if not folder.endswith("_best") else 0  # xxx 作为正样本，xxx_best 作为负样本

                # 遍历 .bpseq 文件
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.bpseq'):
                        bpseq_path = os.path.join(folder_path, file_name)
                        npy_path = os.path.join(self.tmRNA_fm_dir, file_name.replace('.bpseq', '.npy'))

                        if os.path.exists(npy_path):  # 确保 .npy 文件存在
                            self.samples.append((bpseq_path, npy_path, label))


        # 确保 family_dir 是列表
        if not isinstance(self.family_dir, list):
            raise ValueError("family_dir 应该是一个包含多个目录的列表")

        # 遍历所有目录
        for base_dir in self.family_dir:
            if not os.path.isdir(base_dir):
                continue  # 跳过无效目录

            # 遍历目录下的所有文件夹
            for folder in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder)

                if not os.path.isdir(folder_path):
                    continue  # 跳过非文件夹

                # 只处理 xxx（正样本）和 xxx_best（负样本），忽略 xxx_worst
                if folder.endswith("_worst"):
                    continue

                label = 1 if not folder.endswith("_best") else 0  # xxx 作为正样本，xxx_best 作为负样本

                # 遍历 .bpseq 文件
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.bpseq'):
                        bpseq_path = os.path.join(folder_path, file_name)
                        npy_path = os.path.join(self.rna_fm_dir, file_name.replace('.bpseq', '.npy'))

                        if os.path.exists(npy_path):  # 确保 .npy 文件存在
                            self.samples.append((bpseq_path, npy_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_len = 600
        bpseq_path, npy_path, label = self.samples[idx]

        # 读取bpseq数据（这里可以根据需要进行解析）
        seq, edge = self._parse_bpseq(bpseq_path)
        edge_index = torch.LongTensor(edge)

        adj_matrix = torch.zeros((max_len, max_len))
        adj_matrix[edge_index[0], edge_index[1]] = 1
        adj_matrix = adj_matrix.double()


        # 读取RNA-FM数据
        valid_elements = {'A', 'U', 'G', 'C', '0'}
        replaced_data = np.where(np.isin(seq, list(valid_elements)), seq, '0')
        features = encode_onehot(replaced_data)
        features = torch.from_numpy(features)

        rna_fm_data = np.load(npy_path)
        rna_fm_data = rna_fm_data.astype(float)
        rna_fm_data = torch.from_numpy(rna_fm_data)
        x_features = [features, rna_fm_data]
        x_features = torch.cat(x_features, dim=1)
        x_features = x_features.double()

        flatten_x = torch.zeros((max_len, 645))  # 补齐data.x
        flatten_x[:x_features.shape[0], :x_features.shape[1]] = x_features
        flatten_x = flatten_x.double()

        return adj_matrix, flatten_x, label

    def _parse_bpseq(self, bpseq_path):
        sequences = []
        list1 = []
        list2 = []

        with open(bpseq_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                index, base, pair_index = line.strip().split()
                index = int(index) - 1
                pair_index = int(pair_index) - 1
                sequences.append(base)
                if int(index) != -1 and int(pair_index) != -1:
                    list1.append(int(index))
                    list2.append(int(pair_index))
            edge_matrix = np.array([list1, list2])

        return sequences, edge_matrix


def RNA_motif_info_extraction(RNA_seq, RNA_ss):
    # 1. Obtain the index of base pairs.
    pairs_index, pairs_pk_index = parse_vienna_to_pairs(RNA_ss)

    # 2. Construct the index column as well as split the RNA sequence and RNA secondary structure

    RNA_len = len(RNA_seq)
    index_list = list(range(1, RNA_len + 1))

    RNA_seq_split = list(RNA_seq)

    RNA_ss_split = list(RNA_ss)

    # 3. Construct the nt based data frame for one RNA

    motif_info_data_initial = {'index': index_list, 'nt': RNA_seq_split, 'structure': RNA_ss_split}

    motif_info_data_df = pd.DataFrame(data=motif_info_data_initial)

    # 4. Assign the base pair into the motif_info_data_df, determine the kinds of motifs of RNA, and keep the index of motifs

    motif_info_data_df['base_pair_index'] = np.nan
    motif_info_data_df['base_pair_index'] = motif_info_data_df['base_pair_index'].astype(object)

    motif_info_data_df['motif_1'] = np.nan
    motif_info_data_df['motif_1'] = motif_info_data_df['motif_1'].astype(object)

    motif_info_data_df['motif_1_seq'] = np.nan
    motif_info_data_df['motif_1_seq'] = motif_info_data_df['motif_1_seq'].astype(object)

    motif_info_data_df['motif_1_index'] = np.nan
    motif_info_data_df['motif_1_index'] = motif_info_data_df['motif_1_index'].astype(object)

    # 4.1 Determine the types of motifs
    for base_pair in pairs_index:

        motif_info_data_df.at[base_pair[0] - 1, 'base_pair_index'] = base_pair
        motif_info_data_df.at[base_pair[1] - 1, 'base_pair_index'] = base_pair

        # 4.1.1 determine the stack
        if motif_info_data_df.at[base_pair[0], 'structure'] == "(" and \
                motif_info_data_df.at[base_pair[1] - 2, 'structure'] == ")" and \
                [base_pair[0] + 1, base_pair[1] - 1] in pairs_index and \
                not np.isnan(motif_info_data_df.at[base_pair[0] - 1, 'base_pair_index']).any() and \
                not np.isnan(motif_info_data_df.at[base_pair[1] - 1, 'base_pair_index']).any():
            motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] = "stack"
            motif_info_data_df.at[base_pair[1] - 1, 'motif_1'] = "stack"

            motif_1_sequence = motif_info_data_df.at[base_pair[0] - 1, 'nt'] + \
                               motif_info_data_df.at[base_pair[0], 'nt'] + \
                               motif_info_data_df.at[base_pair[1] - 2, 'nt'] + \
                               motif_info_data_df.at[base_pair[1] - 1, 'nt']

            motif_info_data_df.at[base_pair[0] - 1, 'motif_1_seq'] = motif_1_sequence
            motif_info_data_df.at[base_pair[1] - 1, 'motif_1_seq'] = motif_1_sequence
        # 4.1.2 determine the bulge/interior/bifurcation loop and hairpin loop
        elif motif_info_data_df.at[base_pair[0], 'structure'] == "." or \
                motif_info_data_df.at[base_pair[1] - 2, 'structure'] == ".":
            str_within_loop = []
            loop_length = base_pair[1] - base_pair[0]
            for i in range(loop_length - 1):
                str_within_loop.append(motif_info_data_df.at[base_pair[0] + i, 'structure'])

            if "(" in str_within_loop or ")" in str_within_loop:
                motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] = "bul_int_bir_loop"
                motif_info_data_df.at[base_pair[1] - 1, 'motif_1'] = "bul_int_bir_loop"
            else:
                motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] = "hairpin_loop"
                motif_info_data_df.at[base_pair[1] - 1, 'motif_1'] = "hairpin_loop"

                motif_1_sequence = motif_info_data_df.at[base_pair[0] - 1, 'nt']
                for i in range(loop_length):
                    motif_1_sequence += motif_info_data_df.at[base_pair[0] + i, 'nt']

                motif_info_data_df.at[base_pair[0] - 1, 'motif_1_seq'] = motif_1_sequence
                motif_info_data_df.at[base_pair[1] - 1, 'motif_1_seq'] = motif_1_sequence
        # 4.1.3 determine the bifurcation loop
        # else:
        # motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bifurcation_loop"
        # motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bifurcation_loop"

    # 4.2 Determine the index of motifs
    for base_pair in pairs_index:

        # 4.2.1 determine the index of the stack
        if motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] == "stack" and motif_info_data_df.at[
            base_pair[1] - 1, 'motif_1'] == "stack":
            motif_info_data_df.at[base_pair[0] - 1, 'motif_1_index'] = [
                motif_info_data_df.at[base_pair[0] - 1, 'base_pair_index'][0],
                motif_info_data_df.at[base_pair[0], 'base_pair_index'][0],
                motif_info_data_df.at[base_pair[1] - 2, 'base_pair_index'][1],
                motif_info_data_df.at[base_pair[1] - 1, 'base_pair_index'][1]]

            motif_info_data_df.at[base_pair[1] - 1, 'motif_1_index'] = motif_info_data_df.at[
                base_pair[0] - 1, 'motif_1_index']

        # 4.2.2 determine the index of hairpin loop
        elif motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] == "hairpin_loop" and motif_info_data_df.at[
            base_pair[1] - 1, 'motif_1'] == "hairpin_loop":
            loop_index = []
            loop_length = base_pair[1] - base_pair[0]
            for i in range(loop_length + 1):
                loop_index.append(motif_info_data_df.at[base_pair[0] - 1 + i, 'index'])

            motif_info_data_df.at[base_pair[0] - 1, 'motif_1_index'] = loop_index
            motif_info_data_df.at[base_pair[1] - 1, 'motif_1_index'] = motif_info_data_df.at[
                base_pair[0] - 1, 'motif_1_index']

    # 5. Distinguish bifurcation loop from bulge/interior loop

    motif_info_data_df['motif_1_bul_int_bir_loop_base_pairs'] = np.nan
    motif_info_data_df['motif_1_bul_int_bir_loop_base_pairs'] = motif_info_data_df[
        'motif_1_bul_int_bir_loop_base_pairs'].astype(object)

    # 5.1 obtain the index of base pairs whose motif_1 equals to bul_int_bir_loop
    motif_info_data_df_temp = motif_info_data_df[motif_info_data_df["motif_1"] == "bul_int_bir_loop"].reset_index(
        drop=True)

    num_rows_motif_info_data_df_temp = motif_info_data_df_temp.shape[0]

    temp_base_pair_index = []

    for i in range(num_rows_motif_info_data_df_temp):
        if motif_info_data_df_temp.at[i, 'base_pair_index'] not in temp_base_pair_index:
            temp_base_pair_index.append(motif_info_data_df_temp.at[i, 'base_pair_index'])
    '''
    # 5.2 distinguish bifurcation loop from bulge/interior loop  
    for base_pair in temp_base_pair_index:
        loop_length = base_pair[1]-base_pair[0]
        bul_int_bir_loop_motif_1 = []
        for i in range(loop_length-1):
            bul_int_bir_loop_motif_1.append(motif_info_data_df.at[base_pair[0]+i, 'motif_1'])

        if bul_int_bir_loop_motif_1.count("hairpin_loop") >= 4:
            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bifurcation_loop"
            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bifurcation_loop"
        else:
            motif_info_data_df.at[base_pair[0]-1, 'motif_1'] = "bul_int_loop"
            motif_info_data_df.at[base_pair[1]-1, 'motif_1'] = "bul_int_loop"
    '''
    # 5.2.1 distinguish bifurcation loop from bulge/interior loop from number of base pairs point of view

    for base_pair in temp_base_pair_index:
        base_pairs_bul_int_bir_loop = []
        base_pairs_bul_int_bir_loop.append(base_pair)

        loop_length = base_pair[1] - base_pair[0]
        for i in range(loop_length - 1):
            if motif_info_data_df.at[base_pair[0] + i, 'structure'] == "(":
                base_pair_star = motif_info_data_df.at[base_pair[0] + i, 'base_pair_index']
                break

        base_pairs_bul_int_bir_loop.append(base_pair_star)

        structure_indicator = base_pair_star[1]

        while False in motif_info_data_df.loc[structure_indicator:base_pair[1] - 2, ['base_pair_index']].isnull()[
            'base_pair_index'].unique():
            index_base_pair_star_new = motif_info_data_df.loc[structure_indicator:base_pair[1] - 2,
                                       ['base_pair_index']].first_valid_index()
            base_pair_star_new = motif_info_data_df.at[index_base_pair_star_new, 'base_pair_index']

            base_pairs_bul_int_bir_loop.append(base_pair_star_new)
            structure_indicator = base_pair_star_new[1]

        base_pairs_bul_int_bir_loop.sort(key=lambda pair: pair[0])
        # print("The base pairs in this bifurcation/bulge/interior loop is ", base_pairs_bul_int_bir_loop)

        motif_info_data_df.at[base_pair[0] - 1, 'motif_1_bul_int_bir_loop_base_pairs'] = base_pairs_bul_int_bir_loop
        motif_info_data_df.at[base_pair[1] - 1, 'motif_1_bul_int_bir_loop_base_pairs'] = base_pairs_bul_int_bir_loop

        if len(motif_info_data_df.at[base_pair[0] - 1, 'motif_1_bul_int_bir_loop_base_pairs']) == 2:
            motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] = "bul_int_loop"
            motif_info_data_df.at[base_pair[1] - 1, 'motif_1'] = "bul_int_loop"
        elif len(motif_info_data_df.at[base_pair[0] - 1, 'motif_1_bul_int_bir_loop_base_pairs']) >= 3:
            motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] = "bifurcation_loop"
            motif_info_data_df.at[base_pair[1] - 1, 'motif_1'] = "bifurcation_loop"

    # 6. Distinguish the interior loop from bulge loop

    motif_info_data_df['motif_1_bul_int_edge_len'] = 0
    motif_info_data_df['motif_1_bul_int_edge_len'] = motif_info_data_df['motif_1_bul_int_edge_len'].astype(int)

    # 6.1 obtain the index of base pairs whose motif_1 equals to bul_int_loop
    motif_info_data_df_temp = motif_info_data_df[motif_info_data_df["motif_1"] == "bul_int_loop"].reset_index(drop=True)

    num_rows_motif_info_data_df_temp = motif_info_data_df_temp.shape[0]

    temp_base_pair_index = []

    for i in range(num_rows_motif_info_data_df_temp):
        if motif_info_data_df_temp.at[i, 'base_pair_index'] not in temp_base_pair_index:
            temp_base_pair_index.append(motif_info_data_df_temp.at[i, 'base_pair_index'])

    # 6.2 distinguish the interior loop from bulge loop

    for base_pair in temp_base_pair_index:
        # 6.2.1 distinguish the interior loop from bulge loop and obtain the length of the edge of the bulge/interior loop
        loop_length = base_pair[1] - base_pair[0]
        for i in range(loop_length - 1):
            if motif_info_data_df.at[base_pair[0] + i, 'structure'] == "(":
                base_pair_2 = motif_info_data_df.at[base_pair[0] + i, 'base_pair_index']
                break

        if base_pair_2[0] - 1 == base_pair[0] or base_pair[1] - 1 == base_pair_2[1]:
            motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] = "bulge_loop"
            motif_info_data_df.at[base_pair[1] - 1, 'motif_1'] = "bulge_loop"
        else:
            motif_info_data_df.at[base_pair[0] - 1, 'motif_1'] = "interior_loop"
            motif_info_data_df.at[base_pair[1] - 1, 'motif_1'] = "interior_loop"

        motif_info_data_df.at[base_pair[0] - 1, 'motif_1_bul_int_edge_len'] = base_pair_2[0] - base_pair[0] + 1
        motif_info_data_df.at[base_pair[1] - 1, 'motif_1_bul_int_edge_len'] = motif_info_data_df.at[
            base_pair[0] - 1, 'motif_1_bul_int_edge_len']

        # 6.2.2 determine the sequence of interior loop and bulge loop
        loop_length_1 = base_pair_2[0] - base_pair[0]
        loop_length_2 = base_pair[1] - base_pair_2[1]
        motif_1_sequence = motif_info_data_df.at[base_pair[0] - 1, 'nt']

        for i in range(loop_length_1):
            motif_1_sequence += motif_info_data_df.at[base_pair[0] + i, 'nt']

        motif_1_sequence += motif_info_data_df.at[base_pair_2[1] - 1, 'nt']

        for i in range(loop_length_2):
            motif_1_sequence += motif_info_data_df.at[base_pair_2[1] + i, 'nt']

        motif_info_data_df.at[base_pair[0] - 1, 'motif_1_seq'] = motif_1_sequence
        motif_info_data_df.at[base_pair[1] - 1, 'motif_1_seq'] = motif_1_sequence

        # 6.2.3 determine the index of interior loop and bulge loop
        loop_index = []
        for i in range(loop_length_1 + 1):
            loop_index.append(motif_info_data_df.at[base_pair[0] - 1 + i, 'index'])

        for i in range(loop_length_2 + 1):
            loop_index.append(motif_info_data_df.at[base_pair_2[1] - 1 + i, 'index'])

        motif_info_data_df.at[base_pair[0] - 1, 'motif_1_index'] = loop_index
        motif_info_data_df.at[base_pair[1] - 1, 'motif_1_index'] = motif_info_data_df.at[
            base_pair[0] - 1, 'motif_1_index']

    '''    
    # 7. Obtain the motif's info of unpaired "."

    temp_motif_index = []

    for i in range(RNA_len):
        if motif_info_data_df.at[i, 'motif_1_index'] not in temp_motif_index and\
            not np.isnan(motif_info_data_df.at[i, 'motif_1_index']).any() and\
                (motif_info_data_df.at[i, 'motif_1'] == "hairpin_loop" or motif_info_data_df.at[i, 'motif_1'] == "interior_loop" or\
                 motif_info_data_df.at[i, 'motif_1'] == "bulge_loop"):
                    temp_motif_index.append(motif_info_data_df.at[i, 'motif_1_index'])

                    for j in motif_info_data_df.at[i, 'motif_1_index']:
                        if motif_info_data_df.at[j-1, 'structure'] == ".":
                            motif_info_data_df.at[j-1, 'motif_1'] = motif_info_data_df.at[i, 'motif_1']
                            motif_info_data_df.at[j-1, 'motif_1_index'] = motif_info_data_df.at[i, 'motif_1_index']
                            motif_info_data_df.at[j-1, 'motif_1_seq'] = motif_info_data_df.at[i, 'motif_1_seq']
                            motif_info_data_df.at[j-1, 'motif_1_bul_int_edge_len'] = motif_info_data_df.at[i, 'motif_1_bul_int_edge_len']
    '''

    # 8. Obtain the sequence and motif index of bifurcation loop in motif 1

    # 8.1 obtain the index of base pairs whose motif_1 equals to bifurcation_loop

    motif_info_data_df_temp = motif_info_data_df[motif_info_data_df["motif_1"] == "bifurcation_loop"].reset_index(
        drop=True)

    num_rows_motif_info_data_df_temp = motif_info_data_df_temp.shape[0]

    temp_base_pair_index = []

    for i in range(num_rows_motif_info_data_df_temp):
        if motif_info_data_df_temp.at[i, 'base_pair_index'] not in temp_base_pair_index:
            temp_base_pair_index.append(motif_info_data_df_temp.at[i, 'base_pair_index'])

    # 8.2 determine the motif index and sequence of bifurcation loop
    for base_pair in temp_base_pair_index:

        # 8.2.1 obtain the motif index of bifurcation loop
        '''
        base_pairs_bifurcation_loop = []
        base_pairs_bifurcation_loop.append(base_pair)
        loop_index = []
        loop_index.append(motif_info_data_df.at[base_pair[0]-1, 'index'])
        loop_index.append(motif_info_data_df.at[base_pair[1]-1, 'index'])

        loop_length = base_pair[1]-base_pair[0]
        for i in range(loop_length-1):
            if motif_info_data_df.at[base_pair[0]+i, 'structure'] == "(":
                base_pair_star = motif_info_data_df.at[base_pair[0]+i, 'base_pair_index']
                break

        base_pairs_bifurcation_loop.append(base_pair_star)
        loop_index.append(motif_info_data_df.at[base_pair_star[0]-1, 'index'])
        loop_index.append(motif_info_data_df.at[base_pair_star[1]-1, 'index'])

        structure_indicator = base_pair_star[1]

        while False in motif_info_data_df.loc[structure_indicator:base_pair[1]-2, ['base_pair_index']].isnull()['base_pair_index'].unique():
            index_base_pair_star_new = motif_info_data_df.loc[structure_indicator:base_pair[1]-2, ['base_pair_index']].first_valid_index()
            base_pair_star_new = motif_info_data_df.at[index_base_pair_star_new, 'base_pair_index']

            loop_index.append(motif_info_data_df.at[base_pair_star_new[0]-1, 'index'])
            loop_index.append(motif_info_data_df.at[base_pair_star_new[1]-1, 'index'])

            base_pairs_bifurcation_loop.append(base_pair_star_new)
            structure_indicator = base_pair_star_new[1]

        base_pairs_bifurcation_loop.sort(key = lambda pair: pair[0])
        #print("The base pairs in this bifurcation loop is ", base_pairs_bifurcation_loop)
        '''
        base_pairs_bifurcation_loop = motif_info_data_df.at[base_pair[0] - 1, 'motif_1_bul_int_bir_loop_base_pairs']
        # print("The base pairs in this bifurcation loop is ", base_pairs_bifurcation_loop)

        loop_index = [each_nt for each_base_pair in base_pairs_bifurcation_loop for each_nt in each_base_pair]

        edge_length = base_pairs_bifurcation_loop[1][0] - base_pairs_bifurcation_loop[0][0]
        # print("The first edge length is ", edge_length)
        for i in range(edge_length - 1):
            if motif_info_data_df.at[base_pairs_bifurcation_loop[0][0] + i, 'structure'] == ".":
                loop_index.append(motif_info_data_df.at[base_pairs_bifurcation_loop[0][0] + i, 'index'])

        edge_length = base_pairs_bifurcation_loop[0][1] - base_pairs_bifurcation_loop[-1][1]
        # print("The second edge length is ", edge_length)
        for i in range(edge_length - 1):
            if motif_info_data_df.at[base_pairs_bifurcation_loop[-1][1] + i, 'structure'] == ".":
                loop_index.append(motif_info_data_df.at[base_pairs_bifurcation_loop[-1][1] + i, 'index'])

        base_pairs_bifurcation_loop.pop(0)

        length_base_pairs_bifurcation_loop = len(base_pairs_bifurcation_loop)

        for i in range(length_base_pairs_bifurcation_loop - 1):
            edge_length = base_pairs_bifurcation_loop[i + 1][0] - base_pairs_bifurcation_loop[i][1]
            for j in range(edge_length - 1):
                if motif_info_data_df.at[base_pairs_bifurcation_loop[i][1] + j, 'structure'] == ".":
                    loop_index.append(motif_info_data_df.at[base_pairs_bifurcation_loop[i][1] + j, 'index'])

        loop_index = list(map(np.int64, loop_index))
        loop_index.sort()
        # print("The index of nucleotides in this bifurcation loop is ", loop_index)

        motif_info_data_df.at[base_pair[0] - 1, 'motif_1_index'] = loop_index
        motif_info_data_df.at[base_pair[1] - 1, 'motif_1_index'] = motif_info_data_df.at[
            base_pair[0] - 1, 'motif_1_index']

        # 8.2.2 obtain the sequence of bifurcation loop
        len_bifurcation_loop = len(motif_info_data_df.at[base_pair[0] - 1, 'motif_1_index'])
        motif_1_sequence = motif_info_data_df.at[base_pair[0] - 1, 'nt']
        for i in range(1, len_bifurcation_loop):
            motif_1_sequence += motif_info_data_df.at[
                motif_info_data_df.at[base_pair[0] - 1, 'motif_1_index'][i] - 1, 'nt']

        motif_info_data_df.at[base_pair[0] - 1, 'motif_1_seq'] = motif_1_sequence
        motif_info_data_df.at[base_pair[1] - 1, 'motif_1_seq'] = motif_1_sequence

    # 7. Obtain the motif's info of unpaired "."

    temp_motif_index = []

    for i in range(RNA_len):
        if motif_info_data_df.at[i, 'motif_1_index'] not in temp_motif_index and \
                not np.isnan(motif_info_data_df.at[i, 'motif_1_index']).any() and \
                (motif_info_data_df.at[i, 'motif_1'] == "hairpin_loop" or motif_info_data_df.at[
                    i, 'motif_1'] == "interior_loop" or \
                 motif_info_data_df.at[i, 'motif_1'] == "bulge_loop" or motif_info_data_df.at[
                     i, 'motif_1'] == "bifurcation_loop"):
            temp_motif_index.append(motif_info_data_df.at[i, 'motif_1_index'])

            for j in motif_info_data_df.at[i, 'motif_1_index']:
                if motif_info_data_df.at[j - 1, 'structure'] == ".":
                    motif_info_data_df.at[j - 1, 'motif_1'] = motif_info_data_df.at[i, 'motif_1']
                    motif_info_data_df.at[j - 1, 'motif_1_index'] = motif_info_data_df.at[i, 'motif_1_index']
                    motif_info_data_df.at[j - 1, 'motif_1_seq'] = motif_info_data_df.at[i, 'motif_1_seq']
                    motif_info_data_df.at[j - 1, 'motif_1_bul_int_edge_len'] = motif_info_data_df.at[
                        i, 'motif_1_bul_int_edge_len']
                    motif_info_data_df.at[j - 1, 'motif_1_bul_int_bir_loop_base_pairs'] = motif_info_data_df.at[
                        i, 'motif_1_bul_int_bir_loop_base_pairs']

    # 9. Obtain the motif 2 info of each nucleotide

    motif_info_data_df['motif_2'] = np.nan
    motif_info_data_df['motif_2'] = motif_info_data_df['motif_2'].astype(object)

    motif_info_data_df['motif_2_seq'] = np.nan
    motif_info_data_df['motif_2_seq'] = motif_info_data_df['motif_2_seq'].astype(object)

    motif_info_data_df['motif_2_index'] = np.nan
    motif_info_data_df['motif_2_index'] = motif_info_data_df['motif_2_index'].astype(object)

    motif_info_data_df['motif_2_bul_int_edge_len'] = 0
    motif_info_data_df['motif_2_bul_int_edge_len'] = motif_info_data_df['motif_2_bul_int_edge_len'].astype(int)

    temp_motif_index = []

    for i in range(RNA_len):
        if motif_info_data_df.at[i, 'motif_1_index'] not in temp_motif_index and \
                not np.isnan(motif_info_data_df.at[i, 'motif_1_index']).any():
            for j in motif_info_data_df.at[i, 'motif_1_index']:
                if motif_info_data_df.at[j - 1, 'motif_1_index'] != motif_info_data_df.at[i, 'motif_1_index']:
                    motif_info_data_df.at[j - 1, 'motif_2'] = motif_info_data_df.at[i, 'motif_1']
                    motif_info_data_df.at[j - 1, 'motif_2_seq'] = motif_info_data_df.at[i, 'motif_1_seq']
                    motif_info_data_df.at[j - 1, 'motif_2_index'] = motif_info_data_df.at[i, 'motif_1_index']
                    motif_info_data_df.at[j - 1, 'motif_2_bul_int_edge_len'] = motif_info_data_df.at[
                        i, 'motif_1_bul_int_edge_len']

            temp_motif_index.append(motif_info_data_df.at[i, 'motif_1_index'])

    # 10. Calculate the FE of each motif

    motif_info_data_df['motif_1_FE'] = np.nan
    motif_info_data_df['motif_1_FE'] = motif_info_data_df['motif_1_FE'].astype(float)

    motif_info_data_df['motif_2_FE'] = np.nan
    motif_info_data_df['motif_2_FE'] = motif_info_data_df['motif_2_FE'].astype(float)

    for i in range(RNA_len):
        if motif_info_data_df.at[i, 'motif_1'] == "stack":
            motif_info_data_df.at[i, 'motif_1_FE'] = RNA.fold_compound(
                motif_info_data_df.at[i, 'motif_1_seq']).eval_int_loop(i=1, j=4, k=2, l=3)
        elif motif_info_data_df.at[i, 'motif_1'] == "hairpin_loop":
            loop_len = len(motif_info_data_df.at[i, 'motif_1_seq'])
            motif_info_data_df.at[i, 'motif_1_FE'] = RNA.fold_compound(
                motif_info_data_df.at[i, 'motif_1_seq']).eval_hp_loop(i=1, j=loop_len)
        elif motif_info_data_df.at[i, 'motif_1'] == "bulge_loop" or motif_info_data_df.at[
            i, 'motif_1'] == "interior_loop":
            loop_len = len(motif_info_data_df.at[i, 'motif_1_seq'])
            motif_info_data_df.at[i, 'motif_1_FE'] = \
                RNA.fold_compound(motif_info_data_df.at[i, 'motif_1_seq']).eval_int_loop(i=1, j=loop_len,
                                                                                         k=motif_info_data_df.at[
                                                                                             i, 'motif_1_bul_int_edge_len'].item(),
                                                                                         l=motif_info_data_df.at[
                                                                                               i, 'motif_1_bul_int_edge_len'].item() + 1)
        elif motif_info_data_df.at[i, 'motif_1'] == "bifurcation_loop":
            i = motif_info_data_df.at[i, 'motif_1_index'][0].item()
            pt = RNA.ptable(RNA_ss)
            motif_info_data_df.at[i, 'motif_1_FE'] = RNA.fold_compound(RNA_seq).eval_loop_pt(i, pt)
        else:
            motif_info_data_df.at[i, 'motif_1_FE'] = 0

        if motif_info_data_df.at[i, 'motif_2'] == "stack":
            motif_info_data_df.at[i, 'motif_2_FE'] = RNA.fold_compound(
                motif_info_data_df.at[i, 'motif_2_seq']).eval_int_loop(i=1, j=4, k=2, l=3)
        elif motif_info_data_df.at[i, 'motif_2'] == "hairpin_loop":
            loop_len = len(motif_info_data_df.at[i, 'motif_2_seq'])
            motif_info_data_df.at[i, 'motif_2_FE'] = RNA.fold_compound(
                motif_info_data_df.at[i, 'motif_2_seq']).eval_hp_loop(i=1, j=loop_len)
        elif motif_info_data_df.at[i, 'motif_2'] == "bulge_loop" or motif_info_data_df.at[
            i, 'motif_2'] == "interior_loop":
            loop_len = len(motif_info_data_df.at[i, 'motif_2_seq'])
            motif_info_data_df.at[i, 'motif_2_FE'] = \
                RNA.fold_compound(motif_info_data_df.at[i, 'motif_2_seq']).eval_int_loop(i=1, j=loop_len,
                                                                                         k=motif_info_data_df.at[
                                                                                             i, 'motif_2_bul_int_edge_len'].item(),
                                                                                         l=motif_info_data_df.at[
                                                                                               i, 'motif_2_bul_int_edge_len'].item() + 1)
        elif motif_info_data_df.at[i, 'motif_2'] == "bifurcation_loop":
            i = motif_info_data_df.at[i, 'motif_2_index'][0].item()
            pt = RNA.ptable(RNA_ss)
            motif_info_data_df.at[i, 'motif_2_FE'] = RNA.fold_compound(RNA_seq).eval_loop_pt(i, pt)
        else:
            motif_info_data_df.at[i, 'motif_2_FE'] = 0

    del motif_info_data_df["motif_1_bul_int_edge_len"]
    del motif_info_data_df["motif_2_bul_int_edge_len"]
    del motif_info_data_df['motif_1_bul_int_bir_loop_base_pairs']
    return motif_info_data_df


def nt_localized_info_mat_generator(RNA_seq, RNA_ss):
    # 1. Obtain the motif information matrix by utilizing the function RNA_motif_info_extraction
    motif_info_mat = RNA_motif_info_extraction(RNA_seq, RNA_ss)

    motif_info_mat = motif_info_mat[['nt', 'motif_1', 'motif_1_FE', 'motif_2', 'motif_2_FE']]

    # 2. Transform motif_1 and motif_2 to one hot encoding

    # 2.1 construct the transformer
    motif_type_list = ['stack', 'hairpin_loop', 'interior_loop', 'bulge_loop', 'bifurcation_loop', 'NONE']

    # motif_type_list = list(np.array(['stack', 'hairpin_loop', 'interior_loop', 'bulge_loop', 'bifurcation_loop', 'missing_value']).reshape(1,6))

    motif_transform = Pipeline(
        steps=[('tackle_nan', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='NONE')),
               ('motif_ohe', OneHotEncoder(categories=[motif_type_list], sparse=False, handle_unknown='error'))])

    nt_ohe = OneHotEncoder(categories=[['A', 'U', 'C', 'G']], sparse=False, handle_unknown='error')

    data_processor = ColumnTransformer(transformers=[('nt_tf', nt_ohe, ['nt']),
                                                     ('motif_1_tf', motif_transform, ['motif_1']),
                                                     ('motif_2_tf', motif_transform, ['motif_2'])])

    nt_motif_info_mat_np = data_processor.fit_transform(motif_info_mat)

    nt_motif_ohe = pd.DataFrame(nt_motif_info_mat_np)

    nt_motif_ohe.columns = ['A', 'U', 'C', 'G',
                            'stack_motif_1', 'hairpin_loop_motif_1', 'interior_loop_motif_1', 'bulge_loop_motif_1',
                            'bifurcation_loop_motif_1', 'NONE_motif_1',
                            'stack_motif_2', 'hairpin_loop_motif_2', 'interior_loop_motif_2', 'bulge_loop_motif_2',
                            'bifurcation_loop_motif_2', 'NONE_motif_2']

    motif_info_mat = motif_info_mat.join(nt_motif_ohe)
    # motif_info_mat.to_csv("motif_info_mat_test.csv",index=False)

    del motif_info_mat["nt"]
    del motif_info_mat["motif_1"]
    del motif_info_mat["motif_2"]

    motif_info_mat = motif_info_mat.reindex(columns=['A', 'U', 'C', 'G',
                                                     'stack_motif_1', 'hairpin_loop_motif_1', 'interior_loop_motif_1',
                                                     'bulge_loop_motif_1', 'bifurcation_loop_motif_1', 'NONE_motif_1',
                                                     'motif_1_FE',
                                                     'stack_motif_2', 'hairpin_loop_motif_2', 'interior_loop_motif_2',
                                                     'bulge_loop_motif_2', 'bifurcation_loop_motif_2', 'NONE_motif_2',
                                                     'motif_2_FE'])

    motif_info_mat['motif_1_FE'] = motif_info_mat['motif_1_FE'] / 100
    motif_info_mat['motif_2_FE'] = motif_info_mat['motif_2_FE'] / 100

    motif_info_mat['motif_1_FE'] = norm.cdf(motif_info_mat['motif_1_FE'], loc=0, scale=5)
    motif_info_mat['motif_2_FE'] = norm.cdf(motif_info_mat['motif_2_FE'], loc=0, scale=5)

    motif_info_mat_np_arr = motif_info_mat.to_numpy().astype('float64')

    return motif_info_mat_np_arr

def parse_vienna_to_pairs_3d(vienna_structure):
    """
    解析Vienna格式的RNA二级结构字符串并返回配对信息

    参数：
    vienna_structure (str): Vienna格式的RNA二级结构字符串，例如 '..((...))..'

    返回：
    Tuple[List[Tuple[int, int]], List[int]]:
        - 配对位置的列表，[(left1, right1), (left2, right2), ...]
        - 非配对位置的索引列表，例如 [0, 1, 2] 这些位置没有配对
    """
    # 用栈来存储配对的起始位置
    stack = []
    pairs = []
    non_pair_positions = []  # 存储非配对位置
    length = len(vienna_structure)

    # 遍历Vienna格式字符串
    for i, char in enumerate(vienna_structure):
        if char == '(':
            # 左括号表示开始一个新的配对
            stack.append(i)
        elif char == ')':
            # 右括号表示结束一个配对
            if stack:
                left = stack.pop()
                right = i
                pairs.append((left, right))
        elif char == '.':
            # 点表示当前为非配对位置
            non_pair_positions.append(i)

    # 处理配对信息
    pairs_index = pairs  # 配对的位置
    pairs_pk_index = non_pair_positions  # 非配对位置的索引

    return pairs_index, pairs_pk_index


def RNA_matrix_3d_generator_canonical_bp(RNA_seq, RNA_ss):
    #  2.1 obtain the index for the paired bases
    parse_vienna_to_pairs_3d(RNA_ss)
    pairs_index, pairs_pk_index = parse_vienna_to_pairs(RNA_ss)

    # 2.2 construct the colormap matrix
    RNA_seq_split = list(RNA_seq)
    grayscale_mat_c_1 = pd.DataFrame(index=RNA_seq_split, columns=RNA_seq_split)
    grayscale_mat_c_2 = pd.DataFrame(index=RNA_seq_split, columns=RNA_seq_split)
    grayscale_mat_c_3 = pd.DataFrame(index=RNA_seq_split, columns=RNA_seq_split)
    grayscale_mat_c_4 = pd.DataFrame(index=RNA_seq_split, columns=RNA_seq_split)

    # 2.3 assign value to each element of the colormap matrix
    row_name_colorm = list(grayscale_mat_c_1.index)
    col_name_colorm = list(grayscale_mat_c_1.columns)

    # 3.1 In the diagonal, assign vector to each nucleotide
    for i, row_base in enumerate(row_name_colorm, start=1):
        if row_base == 'A':
            grayscale_mat_c_1.iloc[i - 1, i - 1] = 1
            grayscale_mat_c_2.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_3.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_4.iloc[i - 1, i - 1] = 0
        elif row_base == 'U':
            grayscale_mat_c_1.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_2.iloc[i - 1, i - 1] = 1
            grayscale_mat_c_3.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_4.iloc[i - 1, i - 1] = 0
        elif row_base == 'C':
            grayscale_mat_c_1.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_2.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_3.iloc[i - 1, i - 1] = 1
            grayscale_mat_c_4.iloc[i - 1, i - 1] = 0
        elif row_base == 'G':
            grayscale_mat_c_1.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_2.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_3.iloc[i - 1, i - 1] = 0
            grayscale_mat_c_4.iloc[i - 1, i - 1] = 1
        else:
            raise Exception("The nucleotide of this RNA seq has issue.")

    # 3.2 For the paired nucleotides, assign different vectors to represent differnt types of base pairing
    for k, paired_base_index in enumerate(pairs_index, start=1):
        paired_base_index_x = paired_base_index[0] - 1
        paired_base_index_y = paired_base_index[1] - 1

        if row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        '''
        elif row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 2
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 2
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 2
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 2
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        else:
            raise Exception("The base pair of this RNA has issue.")
        '''

    for l, paired_base_index_2 in enumerate(pairs_index, start=1):
        paired_base_index_x = paired_base_index_2[1] - 1
        paired_base_index_y = paired_base_index_2[0] - 1

        if row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        '''
        elif row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 1
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 1
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'A' and col_name_colorm[paired_base_index_y] == 'A':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 2
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'U' and col_name_colorm[paired_base_index_y] == 'U':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 2
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        elif row_name_colorm[paired_base_index_x] == 'G' and col_name_colorm[paired_base_index_y] == 'G':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 2
        elif row_name_colorm[paired_base_index_x] == 'C' and col_name_colorm[paired_base_index_y] == 'C':
            grayscale_mat_c_1.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_2.iloc[paired_base_index_x, paired_base_index_y] = 0
            grayscale_mat_c_3.iloc[paired_base_index_x, paired_base_index_y] = 2
            grayscale_mat_c_4.iloc[paired_base_index_x, paired_base_index_y] = 0
        else:
            raise Exception("The base pair of this RNA has issue.")
        '''

    # 3.3 For the unpaired nucleotides, fill in NAN with 0
    grayscale_mat_c_1 = grayscale_mat_c_1.fillna(0)
    grayscale_mat_c_2 = grayscale_mat_c_2.fillna(0)
    grayscale_mat_c_3 = grayscale_mat_c_3.fillna(0)
    grayscale_mat_c_4 = grayscale_mat_c_4.fillna(0)

    grayscale_mat_c_1_np_arr = grayscale_mat_c_1.to_numpy().astype('int64')
    grayscale_mat_c_2_np_arr = grayscale_mat_c_2.to_numpy().astype('int64')
    grayscale_mat_c_3_np_arr = grayscale_mat_c_3.to_numpy().astype('int64')
    grayscale_mat_c_4_np_arr = grayscale_mat_c_4.to_numpy().astype('int64')

    grayscale_mat = np.array([grayscale_mat_c_1_np_arr, grayscale_mat_c_2_np_arr,
                              grayscale_mat_c_3_np_arr, grayscale_mat_c_4_np_arr])

    return grayscale_mat


def bpseq_to_dot_bracket(bpseq_file):
    """将单个 bpseq 文件转换为 dot-bracket 格式"""
    with open(bpseq_file, 'r') as file:
        lines = file.readlines()

    # 解析 bpseq 内容
    pairs = {}
    sequence_length = 0
    for line in lines:
        if line.strip():
            parts = line.split()
            index = int(parts[0])
            base = parts[1]
            pair = int(parts[2])

            # 记录配对信息
            pairs[index] = pair
            sequence_length = max(sequence_length, index)

    # 构造 dot-bracket 格式
    dot_bracket = ['.'] * sequence_length

    for i in range(1, sequence_length + 1):
        if pairs[i] > i:
            dot_bracket[i - 1] = '('
            dot_bracket[pairs[i] - 1] = ')'

    return ''.join(dot_bracket)


class Nu_res_Dataset(Dataset):
    def __init__(self, bpseq_pos_dir, bpseq_neg_dir):
        self.bpseq_pos_dir = bpseq_pos_dir
        self.bpseq_neg_dir = bpseq_neg_dir


        self.samples = []
        self._load_samples()

    def _load_samples(self):
        # 加载正样本
        for file_name in os.listdir(self.bpseq_pos_dir):
            if file_name.endswith('.bpseq'):
                bpseq_path = os.path.join(self.bpseq_pos_dir, file_name)
                if os.path.exists(bpseq_path):
                    self.samples.append((bpseq_path, 1))  # 1 是正样本标签

        # 加载负样本
        for file_name in os.listdir(self.bpseq_neg_dir):
            if file_name.endswith('.bpseq'):
                bpseq_path = os.path.join(self.bpseq_neg_dir, file_name)
                if os.path.exists(bpseq_path):
                    self.samples.append((bpseq_path, 0))  # 0 是负样本标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_len = 734
        bpseq_path, label = self.samples[idx]

        # 读取bpseq数据（这里可以根据需要进行解析）
        seq, edge = self._parse_bpseq(bpseq_path)
        bracket = bpseq_to_dot_bracket(bpseq_path)
        valid_elements = {'A', 'U', 'G', 'C'}
        valid_list = list(valid_elements)

        # 将 seq 转换为列表，方便修改
        seq_list = list(seq)

        # 遍历序列并随机替换非法字符
        for i in range(len(seq_list)):
            if seq_list[i] not in valid_elements:
                seq_list[i] = random.choice(valid_list)  # 随机替换

        # 重新转换为字符串
        seq = ''.join(seq_list)

        color_mat = RNA_matrix_3d_generator_canonical_bp(seq, bracket)
        nt_localized_mat = nt_localized_info_mat_generator(seq,bracket)
        nt_localized_mat = np.expand_dims(nt_localized_mat, axis=0)
        nt_localized_mat = np.nan_to_num(nt_localized_mat, nan = 0)

        pad_l = max_len - color_mat.shape[1]  # l 维的填充值
        color_mat = np.pad(color_mat, ((0, 0), (0, pad_l), (0, pad_l)), mode='constant', constant_values=0)

        # nt_localized_mat 形状 (1, l, 18)，填充到 (1, max_len, 18)
        pad_l = max_len - nt_localized_mat.shape[1]
        nt_localized_mat = np.pad(nt_localized_mat, ((0, 0), (0, pad_l), (0, 0)), mode='constant', constant_values=0)

        color_mat = torch.from_numpy(color_mat)
        color_mat = color_mat.type(torch.float)

        nt_localized_mat = torch.from_numpy(nt_localized_mat)
        nt_localized_mat = nt_localized_mat.type(torch.float)

        return color_mat, nt_localized_mat, label

    def _parse_bpseq(self, bpseq_path):
        sequences = []
        list1 = []
        list2 = []

        with open(bpseq_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                index, base, pair_index = line.strip().split()
                index = int(index) - 1
                pair_index = int(pair_index) - 1
                sequences.append(base)
                if int(index) != -1 and int(pair_index) != -1:
                    list1.append(int(index))
                    list2.append(int(pair_index))
            edge_matrix = np.array([list1, list2])

        return sequences, edge_matrix


class RNA_family_Dataset_for_nu(Dataset):
    def __init__(self, family_dir,tmRNA_dir):
        self.family_dir = family_dir
        self.tmRNA_dir = tmRNA_dir

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        if os.path.exists(self.tmRNA_dir):
            # 遍历 base_dir 目录下的所有文件夹
            for folder in os.listdir(self.tmRNA_dir):
                folder_path = os.path.join(self.tmRNA_dir, folder)

                if not os.path.isdir(folder_path):
                    continue  # 跳过非文件夹

                # 只处理 xxx（正样本）和 xxx_best（负样本），忽略 xxx_worst
                if folder.endswith("_best"):
                    continue

                label = 1 if not folder.endswith("_worst") else 0  # xxx 作为正样本，xxx_best 作为负样本

                # 遍历 .bpseq 文件
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.bpseq'):
                        bpseq_path = os.path.join(folder_path, file_name)

                        if os.path.exists(bpseq_path):  # 确保 .npy 文件存在
                            self.samples.append((bpseq_path, label))

        # 确保 family_dir 是列表
        if not isinstance(self.family_dir, list):
            raise ValueError("family_dir 应该是一个包含多个目录的列表")

        # 遍历所有目录
        for base_dir in self.family_dir:
            if not os.path.isdir(base_dir):
                continue  # 跳过无效目录

            # 遍历目录下的所有文件夹
            for folder in os.listdir(base_dir):
                folder_path = os.path.join(base_dir, folder)

                if not os.path.isdir(folder_path):
                    continue  # 跳过非文件夹

                # 只处理 xxx（正样本）和 xxx_best（负样本），_xxx表示忽略_xxx
                if folder.endswith("_best"):
                    continue

                label = 1 if not folder.endswith("_worst") else 0

                # 遍历 .bpseq 文件
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.bpseq'):
                        bpseq_path = os.path.join(folder_path, file_name)

                        if os.path.exists(bpseq_path):  # 确保 .npy 文件存在
                            self.samples.append((bpseq_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_len = 600
        bpseq_path, label = self.samples[idx]

        # 读取bpseq数据（这里可以根据需要进行解析）
        seq, edge = self._parse_bpseq(bpseq_path)
        bracket = bpseq_to_dot_bracket(bpseq_path)
        valid_elements = {'A', 'U', 'G', 'C'}
        valid_list = list(valid_elements)

        # 将 seq 转换为列表，方便修改
        seq_list = list(seq)

        # 遍历序列并随机替换非法字符
        for i in range(len(seq_list)):
            if seq_list[i] not in valid_elements:
                seq_list[i] = random.choice(valid_list)  # 随机替换

        # 重新转换为字符串
        seq = ''.join(seq_list)

        color_mat = RNA_matrix_3d_generator_canonical_bp(seq, bracket)
        nt_localized_mat = nt_localized_info_mat_generator(seq, bracket)
        nt_localized_mat = np.expand_dims(nt_localized_mat, axis=0)
        nt_localized_mat = np.nan_to_num(nt_localized_mat, nan=0)

        pad_l = max_len - color_mat.shape[1]  # l 维的填充值
        color_mat = np.pad(color_mat, ((0, 0), (0, pad_l), (0, pad_l)), mode='constant', constant_values=0)

        # nt_localized_mat 形状 (1, l, 18)，填充到 (1, max_len, 18)
        pad_l = max_len - nt_localized_mat.shape[1]
        nt_localized_mat = np.pad(nt_localized_mat, ((0, 0), (0, pad_l), (0, 0)), mode='constant', constant_values=0)

        color_mat = torch.from_numpy(color_mat)
        color_mat = color_mat.type(torch.float)

        nt_localized_mat = torch.from_numpy(nt_localized_mat)
        nt_localized_mat = nt_localized_mat.type(torch.float)

        return color_mat, nt_localized_mat, label

    def _parse_bpseq(self, bpseq_path):
        sequences = []
        list1 = []
        list2 = []

        with open(bpseq_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                index, base, pair_index = line.strip().split()
                index = int(index) - 1
                pair_index = int(pair_index) - 1
                sequences.append(base)
                if int(index) != -1 and int(pair_index) != -1:
                    list1.append(int(index))
                    list2.append(int(pair_index))
            edge_matrix = np.array([list1, list2])

        return sequences, edge_matrix


class RNA_ent_Dataset(Dataset):
    def __init__(self, fm_pos_dir, fm_neg_dir, rna_str_dir, seq_pos_dir, seq_neg_dir):
        self.seq_pos_dir = seq_pos_dir
        self.seq_neg_dir = seq_neg_dir
        self.fm_pos_dir = fm_pos_dir
        self.fm_neg_dir = fm_neg_dir
        self.rna_str_dir = rna_str_dir

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        # 加载正样本
        for file_name in os.listdir(self.fm_pos_dir):
            if file_name.endswith('.npy'):
                npy_path = os.path.join(self.fm_pos_dir, file_name)
                seq_path = os.path.join(self.seq_pos_dir, file_name.replace('.npy', '.fasta'))
                bpseq_path = os.path.join(self.rna_str_dir, file_name.replace('.npy', '.bpseq'))
                if os.path.exists(bpseq_path):
                    self.samples.append((bpseq_path, seq_path, npy_path, 1))  # 1 是正样本标签

        # 加载负样本
        for file_name in os.listdir(self.fm_neg_dir):
            if file_name.endswith('.npy'):
                npy_path = os.path.join(self.fm_neg_dir, file_name)
                seq_path = os.path.join(self.seq_pos_dir, file_name.replace('.npy', '.fasta'))
                bpseq_path = os.path.join(self.rna_str_dir, file_name.replace('.npy', '.bpseq'))
                if os.path.exists(bpseq_path):
                    self.samples.append((bpseq_path, seq_path, npy_path, 0))  # 0 是负数样本标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_len = 696
        bpseq_path, seq_path, npy_path, label = self.samples[idx]

        # 读取bpseq数据（这里可以根据需要进行解析）
        _, edge = self._parse_bpseq(bpseq_path)
        edge_index = torch.LongTensor(edge)

        with open(seq_path, 'r') as f:
            lines = f.readlines()
            seq = ''.join([line.strip() for line in lines if not line.startswith('>')]).upper()
            seq = list(seq)

        adj_matrix = torch.zeros((max_len, max_len))
        adj_matrix[edge_index[0], edge_index[1]] = 1
        adj_matrix = adj_matrix.double()


        # 读取RNA-FM数据
        valid_elements = {'A', 'U', 'G', 'C', '0'}
        replaced_data = np.where(np.isin(seq, list(valid_elements)), seq, '0')
        features = encode_onehot(replaced_data)
        features = torch.from_numpy(features)

        rna_fm_data = np.load(npy_path)
        rna_fm_data = rna_fm_data.astype(float)
        rna_fm_data = torch.from_numpy(rna_fm_data)
        x_features = [features, rna_fm_data]
        x_features = torch.cat(x_features, dim=1)
        x_features = x_features.double()

        flatten_x = torch.zeros((max_len, 645))  # 补齐data.x
        flatten_x[:x_features.shape[0], :x_features.shape[1]] = x_features
        flatten_x = flatten_x.double()

        return adj_matrix, flatten_x, label

    def _parse_bpseq(self, bpseq_path):
        sequences = []
        list1 = []
        list2 = []

        with open(bpseq_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                index, base, pair_index = line.strip().split()
                index = int(index) - 1
                pair_index = int(pair_index) - 1
                sequences.append(base)
                if int(index) != -1 and int(pair_index) != -1:
                    list1.append(int(index))
                    list2.append(int(pair_index))
            edge_matrix = np.array([list1, list2])

        return sequences, edge_matrix


class RNA_ent_Dataset_for_nu(Dataset):
    def __init__(self, seq_pos_dir, seq_neg_dir, rna_str_dir):
        self.seq_pos_dir = seq_pos_dir
        self.seq_neg_dir = seq_neg_dir
        self.rna_str_dir = rna_str_dir

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        # 加载正样本
        for file_name in os.listdir(self.seq_pos_dir):
            if file_name.endswith('.fasta'):
                seq_path = os.path.join(self.seq_pos_dir, file_name)
                bpseq_path = os.path.join(self.rna_str_dir, file_name.replace('.fasta', '.bpseq'))
                if os.path.exists(bpseq_path):
                    self.samples.append((bpseq_path, seq_path, 1))  # 1 是正样本标签

        # 加载负样本
        for file_name in os.listdir(self.seq_neg_dir):
            if file_name.endswith('.fasta'):
                seq_path = os.path.join(self.seq_neg_dir, file_name)
                bpseq_path = os.path.join(self.rna_str_dir, file_name.replace('.fasta', '.bpseq'))
                if os.path.exists(bpseq_path):
                    self.samples.append((bpseq_path, seq_path, 0))  # 0 是负数样本标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_len = 696
        bpseq_path, seq_path, label = self.samples[idx]

        # 读取bpseq数据（这里可以根据需要进行解析）
        _, edge = self._parse_bpseq(bpseq_path)
        bracket = bpseq_to_dot_bracket(bpseq_path)
        valid_elements = {'A', 'U', 'G', 'C'}
        valid_list = list(valid_elements)

        with open(seq_path, 'r') as f:
            lines = f.readlines()
            seq = ''.join([line.strip() for line in lines if not line.startswith('>')])

        # 将 seq 转换为列表，方便修改
        seq_list = list(seq)

        # 遍历序列并随机替换非法字符
        for i in range(len(seq_list)):
            if seq_list[i] not in valid_elements:
                seq_list[i] = random.choice(valid_list)  # 随机替换

        # 重新转换为字符串
        seq = ''.join(seq_list)

        color_mat = RNA_matrix_3d_generator_canonical_bp(seq, bracket)
        nt_localized_mat = nt_localized_info_mat_generator(seq, bracket)
        nt_localized_mat = np.expand_dims(nt_localized_mat, axis=0)
        nt_localized_mat = np.nan_to_num(nt_localized_mat, nan=0)

        pad_l = max_len - color_mat.shape[1]  # l 维的填充值
        color_mat = np.pad(color_mat, ((0, 0), (0, pad_l), (0, pad_l)), mode='constant', constant_values=0)

        # nt_localized_mat 形状 (1, l, 18)，填充到 (1, max_len, 18)
        pad_l = max_len - nt_localized_mat.shape[1]
        nt_localized_mat = np.pad(nt_localized_mat, ((0, 0), (0, pad_l), (0, 0)), mode='constant', constant_values=0)

        color_mat = torch.from_numpy(color_mat)
        color_mat = color_mat.type(torch.float)

        nt_localized_mat = torch.from_numpy(nt_localized_mat)
        nt_localized_mat = nt_localized_mat.type(torch.float)

        return color_mat, nt_localized_mat, label

    def _parse_bpseq(self, bpseq_path):
        sequences = []
        list1 = []
        list2 = []

        with open(bpseq_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                index, base, pair_index = line.strip().split()
                index = int(index) - 1
                pair_index = int(pair_index) - 1
                sequences.append(base)
                if int(index) != -1 and int(pair_index) != -1:
                    list1.append(int(index))
                    list2.append(int(pair_index))
            edge_matrix = np.array([list1, list2])

        return sequences, edge_matrix


class ClipDataset(Dataset):
    def __init__(self, bpseq_pos_dir, rna_fm_dir):
        self.bpseq_pos_dir = bpseq_pos_dir
        self.rna_fm_dir = rna_fm_dir

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        # 加载正样本
        for file_name in os.listdir(self.bpseq_pos_dir):
            if file_name.endswith('.bpseq'):
                bpseq_path = os.path.join(self.bpseq_pos_dir, file_name)
                npy_path = os.path.join(self.rna_fm_dir, file_name.replace('.bpseq', '.npy'))
                if os.path.exists(npy_path):
                    self.samples.append((bpseq_path, npy_path, 1))  # 1 是正样本标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_len = 734# 734  256
        bpseq_path, npy_path, label = self.samples[idx]

        # 读取bpseq数据（这里可以根据需要进行解析）
        seq, edge = self._parse_bpseq(bpseq_path)
        edge_index = torch.LongTensor(edge)

        adj_matrix = torch.zeros((max_len, max_len))
        adj_matrix[edge_index[0], edge_index[1]] = 1
        adj_matrix = adj_matrix.double()


        # 读取RNA-FM数据
        valid_elements = {'A', 'U', 'G', 'C', '0'}
        replaced_data = np.where(np.isin(seq, list(valid_elements)), seq, '0')
        features = encode_onehot(replaced_data)
        features = torch.from_numpy(features)

        rna_fm_data = np.load(npy_path)
        rna_fm_data = rna_fm_data.astype(float)
        rna_fm_data = torch.from_numpy(rna_fm_data)
        x_features = [features, rna_fm_data]
        x_features = torch.cat(x_features, dim=1)
        x_features = x_features.double()

        flatten_x = torch.zeros((max_len, 645))  # 补齐data.x
        flatten_x[:x_features.shape[0], :x_features.shape[1]] = x_features
        flatten_x = flatten_x.double()

        return adj_matrix, flatten_x

    def _parse_bpseq(self, bpseq_path):
        sequences = []
        list1 = []
        list2 = []

        with open(bpseq_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                index, base, pair_index = line.strip().split()
                index = int(index) - 1
                pair_index = int(pair_index) - 1
                sequences.append(base)
                if int(index) != -1 and int(pair_index) != -1:
                    list1.append(int(index))
                    list2.append(int(pair_index))
            edge_matrix = np.array([list1, list2])

        return sequences, edge_matrix



# 示例使用
bpseq_pos_dir = "/home/Shihx/dataset/TestSetA"
bpseq_neg_dir = "/home/Shihx/dataset/pre_Atestset"
rna_fm_dir = "/home/Shihx/RNA-FM/redevelop/TestA/representations"

# dataset = RNASeqDataset(bpseq_pos_dir, bpseq_neg_dir, rna_fm_dir)
# print(len(dataset))
# print(dataset[0])
# dataset = Nu_res_Dataset(bpseq_pos_dir, bpseq_neg_dir)
# print(len(dataset))
# print(dataset[0])

family_dir = ["/home/Shihx/dataset_diff_family/RNAStrAlign/16S_rRNA_database",
              "/home/Shihx/dataset_diff_family/RNAStrAlign/group_I_intron_database",
              "/home/Shihx/dataset_diff_family/RNAStrAlign/RNaseP_database",
              "/home/Shihx/dataset_diff_family/RNAStrAlign/SRP_database",
              "/home/Shihx/dataset_diff_family/RNAStrAlign/telomerase",
              ]
rna_fm_damily_dir = r"/home/Shihx/RNA-FM/redevelop/tRNA_rRNA_other_family/representations"

# dataset = RNA_family_Dataset(family_dir, rna_fm_damily_dir)
# print(len(dataset))
# print(dataset[0])

# dataset = RNA_ent_Dataset(seq_pos_dir="/home/Shihx/ENTdataset/free_test_real",
#                               seq_neg_dir="/home/Shihx/ENTdataset/free_test_worst",
#                               fm_pos_dir="/home/Shihx/RNA-FM/redevelop/NU_resnet/representations",
#                               fm_neg_dir="/home/Shihx/RNA-FM/redevelop/NU_resnet_generated/representations",
#                               rna_str_dir="/home/Shihx/ENTdataset/free_test_struc")
# print(len(dataset))
# print(dataset[0])
