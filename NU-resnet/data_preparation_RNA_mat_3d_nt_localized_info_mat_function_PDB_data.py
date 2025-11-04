
import numpy as np
import pandas as pd
from RNA_matrix_3d import RNA_matrix_3d_generator_canonical_bp
from nt_localized_info_matrix_generation_function import nt_localized_info_mat_generator


def encode_onehot(input):
    list_RNA = ['A', 'U', 'G', 'C']
    classes = set(list_RNA)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, input)),
                             dtype=np.int32)
    return labels_onehot

def create_fixed_adjacency_matrix(structure, maxlen=410):
    """
    根据点括号字符串创建一个固定大小的二维矩阵，
    不足部分用零填充，矩阵大小为 maxlen * maxlen。

    Args:
        structure (str): RNA二级结构的点括号字符串。
        maxlen (int): 矩阵的固定大小，默认410。

    Returns:
        adjacency_matrix (np.ndarray): 碱基配对关系的固定大小矩阵。
    """
    n = len(structure)
    if n > maxlen:
        raise ValueError(f"结构长度 {n} 超过了指定的最大长度 {maxlen}！")

    # 初始化全零矩阵
    adjacency_matrix = np.zeros((n, n), dtype=int)
    stack = []  # 用于存储未匹配的左括号位置

    # 遍历点括号字符串
    for idx, char in enumerate(structure):
        if char == '(':  # 左括号
            stack.append(idx)
        elif char == ')':  # 右括号
            if stack:
                left_idx = stack.pop()
                # 更新矩阵对应位置
                adjacency_matrix[left_idx-1, idx-1] = 1
                adjacency_matrix[idx-1, left_idx-1] = 1  # 对称性

    # 使用 np.pad 将矩阵补零至固定大小
    padded_matrix = np.pad(adjacency_matrix, pad_width=((0, maxlen - n), (0, maxlen - n)), mode='constant', constant_values=0)

    padded_matrix = np.expand_dims(padded_matrix, axis=0)

    return padded_matrix


def data_prep_RNA_mat_3d_nt_localized_info_mat_PDB_data(Original_RNA_Data, padding_length,
                                                        embedding_folder = r'data/fm_representations/NU_resnet',
                                                        embedding_generated_folder = r"data/fm_representations/NU_resnet_generated"):
    
    # 1.1.1 Shuffle the data
    Original_RNA_Data = Original_RNA_Data.sample(frac=1, replace=False, random_state=7).reset_index(drop=True)
    
    # 1. Create columns and obtain the shape of the Original_RNA_Data
    #Original_RNA_Data['Normalized_MCC']=np.nan
    Original_RNA_Data['color_matrix_seq_correctss']=np.nan
    Original_RNA_Data['color_matrix_seq_exprtss']=np.nan
    Original_RNA_Data['nt_localized_info_matrix_seq_correctss']=np.nan
    Original_RNA_Data['nt_localized_info_matrix_seq_exprtss']=np.nan
    Original_RNA_Data['embedding'] = np.nan  # 大语言模型嵌入
    Original_RNA_Data['RNA_ss'] = np.nan
    Original_RNA_Data['embedding_generated'] = np.nan

    Original_RNA_Data['color_matrix_seq_correctss'] = Original_RNA_Data['color_matrix_seq_correctss'].astype(object)
    Original_RNA_Data['color_matrix_seq_exprtss'] = Original_RNA_Data['color_matrix_seq_exprtss'].astype(object)
    
    Original_RNA_Data['nt_localized_info_matrix_seq_correctss'] = Original_RNA_Data['nt_localized_info_matrix_seq_correctss'].astype(object)
    Original_RNA_Data['nt_localized_info_matrix_seq_exprtss'] = Original_RNA_Data['nt_localized_info_matrix_seq_exprtss'].astype(object)

    Original_RNA_Data['embedding_generated'] = Original_RNA_Data['embedding_generated'].astype(object)
    Original_RNA_Data['embedding'] = Original_RNA_Data['embedding'].astype(object)
    Original_RNA_Data['RNA_ss'] = Original_RNA_Data['RNA_ss'].astype(object)
    
    # 2.1 Obtain the number of rows and columns of the data frame
    num_row_Original_RNA_Data = Original_RNA_Data.shape[0]
    num_col_Original_RNA_Data = Original_RNA_Data.shape[1]

    def load_and_pad_embedding(rna_name, seq, folder, max_length=410):
        embedding_path = f"{folder}/{rna_name}.npy"
        embedding = np.load(embedding_path)
        L, D = embedding.shape
        onehot = encode_onehot(seq)
        L_onehot, D_onehot = onehot.shape

        # 检查是否需要补零
        if L < max_length:
            padded_embedding = np.pad(embedding, ((0, max_length - L), (0, 0)), mode='constant')
            padded_onehot = np.pad(onehot, ((0, max_length - L_onehot), (0, 0)), mode='constant')
        else:
            padded_embedding = embedding[:max_length, :]
            padded_onehot = onehot[:max_length, :]

        # 拼接嵌入和One-hot编码
        combined_embedding = np.concatenate((padded_embedding, padded_onehot), axis=1)

        return combined_embedding
    
    # 2.2 Genarate a grayscale color matrix and nt localized info matrix for each pair of RNA Sequence and RNA Secondary Structure
    for i in range(num_row_Original_RNA_Data):
        Original_RNA_Data.at[i, 'RNA_ss'] = create_fixed_adjacency_matrix(Original_RNA_Data.at[i,'RNA_secondary_structure'])
        Original_RNA_Data.at[i, 'embedding'] = load_and_pad_embedding(Original_RNA_Data.at[i, 'RNA_name'], Original_RNA_Data.at[i, 'RNA_seq_upper'], embedding_folder)
        Original_RNA_Data.at[i, 'embedding_generated'] = load_and_pad_embedding(Original_RNA_Data.at[i, 'RNA_name'], Original_RNA_Data.at[i, 'RNA_seq_upper'], embedding_generated_folder)
        Original_RNA_Data.at[i, 'color_matrix_seq_correctss']=RNA_matrix_3d_generator_canonical_bp(Original_RNA_Data.at[i, 'RNA_seq_upper'], Original_RNA_Data.at[i, 'RNA_secondary_structure'])
        Original_RNA_Data.at[i, 'color_matrix_seq_exprtss']=RNA_matrix_3d_generator_canonical_bp(Original_RNA_Data.at[i, 'RNA_generated_seq'], Original_RNA_Data.at[i, 'RNA_secondary_structure'])
        Original_RNA_Data.at[i, 'nt_localized_info_matrix_seq_correctss']=nt_localized_info_mat_generator(Original_RNA_Data.at[i, 'RNA_seq_upper'], Original_RNA_Data.at[i, 'RNA_secondary_structure'])
        Original_RNA_Data.at[i, 'nt_localized_info_matrix_seq_exprtss']=nt_localized_info_mat_generator(Original_RNA_Data.at[i, 'RNA_generated_seq'], Original_RNA_Data.at[i, 'RNA_secondary_structure'])
    
    # 2.3 Normalize the MCC
    #Original_RNA_Data['Normalized_MCC'] = (Original_RNA_Data['MCC']+1)/2
    
    # For each RNA sample, the code generates two grayscale color matrix and two nt localized info matrix. One grayscale color matrix and one nt localized info matrix are from actual RNA secondary structure. Another
    # one grayscale color matrix and another one nt localized info matrix are from predicted RNA secondary structure.  
    
    # 2.3.1.1 Obtain the grayscale color matrix and nt localized info matrix from the actual RNA secondary structure
    color_mat_3d_actual_RNA_structure = Original_RNA_Data[['color_matrix_seq_correctss', 'nt_localized_info_matrix_seq_correctss', 'embedding','RNA_ss']].copy(deep=True)
    
    # 2.3.1.2 Add one column normalized MCC score for the corresponding grayscale color matrix 
    color_mat_3d_actual_RNA_structure['dist_positive_negative_samples']=np.nan
    color_mat_3d_actual_RNA_structure['dist_positive_negative_samples']=0
    
    # 2.3.2.1 Obtain the grayscale color matrix and nt localized info matrix from the predicted RNA secondary structure
    color_mat_3d_predicted_RNA_structure = Original_RNA_Data[['color_matrix_seq_exprtss', 'nt_localized_info_matrix_seq_exprtss', 'dist_positive_negative_samples','embedding_generated','RNA_ss']].copy(deep=True)
    
    # 2.3.2.2 Keep the grayscale color matrix whose corresponding normalized MCC is less than 1
    color_mat_3d_predicted_RNA_structure = color_mat_3d_predicted_RNA_structure[color_mat_3d_predicted_RNA_structure['dist_positive_negative_samples'] > 0]
    
    # 2.3.3 Rename the corresponding column name
    color_mat_3d_actual_RNA_structure = color_mat_3d_actual_RNA_structure.rename(columns={"color_matrix_seq_correctss":"color_matrix_utilized", "nt_localized_info_matrix_seq_correctss":"nt_localized_info_matrix_utilized"})
    
    color_mat_3d_predicted_RNA_structure = color_mat_3d_predicted_RNA_structure.rename(columns={"color_matrix_seq_exprtss":"color_matrix_utilized", "nt_localized_info_matrix_seq_exprtss":"nt_localized_info_matrix_utilized","embedding_generated":"embedding"})
    
    # 2.3.4 Combine two data sets
    color_matrix_data_set_combined = pd.concat([color_mat_3d_actual_RNA_structure, color_mat_3d_predicted_RNA_structure], axis=0, ignore_index=True)
    
    # 2.5 Create the label column: 1 represents the positive sample and 0 represents the negative sample
    color_matrix_data_set_combined['RNA_Label'] = np.where(color_matrix_data_set_combined['dist_positive_negative_samples'] > 0, 0, 1)
    
    # 2.6 Obtain the number of rows in the color_matrix_data_set_combined
    num_row_color_matrix_data_set_combined = color_matrix_data_set_combined.shape[0]
    
    # 3.1 Obatin the size of each grayscale color matrix
    color_matrix_data_set_combined['color_mat_size'] = np.nan
    
    for i in range(num_row_color_matrix_data_set_combined):
        color_matrix_data_set_combined.at[i, 'color_mat_size'] = color_matrix_data_set_combined.iloc[i,0].shape[2]
    
    # Obtain the maximum size of all grayscale color matrix and do the padding for all grayscale color matrix and all nt localized info matrix
    size_maximum = padding_length
    
    print("The maximum length of the RNA sequences is {}.".format(size_maximum))
    
    color_matrix_data_set_combined['color_matrix_utilized_for_CNN_padding'] = np.nan
    color_matrix_data_set_combined['color_matrix_utilized_for_CNN_padding'] = color_matrix_data_set_combined['color_matrix_utilized_for_CNN_padding'].astype(object)
    color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'] = np.nan
    color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'] = color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'].astype(object)
    
    for i in range(num_row_color_matrix_data_set_combined):
        size_diff = size_maximum - color_matrix_data_set_combined.at[i, 'color_mat_size']
        size_diff_utilized = int(size_diff)
        color_matrix_data_set_combined.at[i, 'nt_localized_info_matrix_utilized_for_CNN_padding']=np.pad(color_matrix_data_set_combined.iloc[i,1], ((0, size_diff_utilized), (0, 0)), 'constant')
        
        if (size_diff % 2) == 0:
            num_padding_left_above = int(size_diff/2)
            num_padding_right_below = int(size_diff/2)
        else:
            num_padding_left_above = int(size_diff/2)
            num_padding_right_below = int(size_diff/2)+1
        
        color_matrix_data_set_combined.at[i, 'color_matrix_utilized_for_CNN_padding']=np.pad(color_matrix_data_set_combined.iloc[i,0], ((0, 0), (num_padding_left_above, num_padding_right_below), (num_padding_left_above, num_padding_right_below)), 'constant')
    
    # 3.2 Test the size of the grayscale color matrix after do the padding
    color_matrix_data_set_combined['color_mat_size_test_af_pad'] = np.nan
    color_matrix_data_set_combined['nt_localized_info_mat_size_test_af_pad'] = np.nan
    
    for i in range(num_row_color_matrix_data_set_combined):
        color_matrix_data_set_combined.at[i, 'color_mat_size_test_af_pad'] = color_matrix_data_set_combined.at[i, 'color_matrix_utilized_for_CNN_padding'].shape[2]
        color_matrix_data_set_combined.at[i, 'nt_localized_info_mat_size_test_af_pad'] = color_matrix_data_set_combined.at[i, 'nt_localized_info_matrix_utilized_for_CNN_padding'].shape[0]
    
    # 3.3 Shuffle the data
    color_matrix_data_set_combined = color_matrix_data_set_combined.sample(frac=1, random_state=59).reset_index(drop=True)
    
    # 4.1 Obtain the grayscale color matrix and label for each RNA samples. And transfer the df to np
    x = color_matrix_data_set_combined[['color_matrix_utilized_for_CNN_padding', 'nt_localized_info_matrix_utilized_for_CNN_padding','embedding','RNA_ss']]
    y = color_matrix_data_set_combined[['RNA_Label']]
    
    train_x_color_mat = x.loc[:, 'color_matrix_utilized_for_CNN_padding'].to_numpy()
    
    train_x_nt_localized_info_mat = x.loc[:, 'nt_localized_info_matrix_utilized_for_CNN_padding'].to_numpy()

    train_x_embedding = x.loc[:, 'embedding'].to_numpy()

    train_x_ss = x.loc[:, 'RNA_ss'].to_numpy()
    
    train_y = y.iloc[:,0].to_numpy()
    
    # 4.2 Transfer the 1-D np array to 3-D np array
    train_x_color_mat_stack = train_x_color_mat[0][np.newaxis,:,:,:]
    
    for j, mat in enumerate(train_x_color_mat[1:]):
        color_mat_augmented = mat[np.newaxis,:,:,:]
        train_x_color_mat_stack = np.vstack((train_x_color_mat_stack, color_mat_augmented))
    
    train_x_nt_localized_info_mat_stack = train_x_nt_localized_info_mat[0][np.newaxis,np.newaxis,:,:]
        
    for j, mat in enumerate(train_x_nt_localized_info_mat[1:]):
        nt_localized_info_mat_augmented = mat[np.newaxis,np.newaxis,:,:]
        train_x_nt_localized_info_mat_stack = np.vstack((train_x_nt_localized_info_mat_stack, nt_localized_info_mat_augmented))


    train_x_embedding_stack = train_x_embedding[0][np.newaxis, :, :]  # 增加一个新的维度，使其符合需要的形状

    for j, mat in enumerate(train_x_embedding[1:]):
        embedding_mat_augmented = mat[np.newaxis, :, :]  # 给每个矩阵增加一个新的维度
        train_x_embedding_stack = np.vstack((train_x_embedding_stack, embedding_mat_augmented))  # 合并到栈中

    # 处理 train_x_ss
    train_x_ss_stack = train_x_ss[0][np.newaxis, :, :, :]  # 增加一个新的维度，使其符合需要的形状

    for j, mat in enumerate(train_x_ss[1:]):
        ss_mat_augmented = mat[np.newaxis, :, :, :]  # 给每个矩阵增加一个新的维度
        train_x_ss_stack = np.vstack((train_x_ss_stack, ss_mat_augmented))  # 合并到栈中

    return train_x_color_mat_stack, train_x_nt_localized_info_mat_stack, train_y, train_x_embedding_stack, train_x_ss_stack


def data_prep_RNA_for_tRNA_rRNA(Original_RNA_Data, padding_length,
                                                        embedding_folder=r'data/fm_representations/RNA_families',
                                                        embedding_generated_folder=r"data/fm_representations/RNA_families_generated"):
    # 1.1.1 Shuffle the data
    Original_RNA_Data = Original_RNA_Data.sample(frac=1, replace=False, random_state=7).reset_index(drop=True)

    # 1. Create columns and obtain the shape of the Original_RNA_Data
    # Original_RNA_Data['Normalized_MCC']=np.nan
    Original_RNA_Data['color_matrix_seq_correctss'] = np.nan
    Original_RNA_Data['color_matrix_seq_exprtss'] = np.nan
    Original_RNA_Data['nt_localized_info_matrix_seq_correctss'] = np.nan
    Original_RNA_Data['nt_localized_info_matrix_seq_exprtss'] = np.nan
    Original_RNA_Data['embedding'] = np.nan  # 大语言模型嵌入
    Original_RNA_Data['RNA_ss'] = np.nan
    Original_RNA_Data['embedding_generated'] = np.nan

    Original_RNA_Data['color_matrix_seq_correctss'] = Original_RNA_Data['color_matrix_seq_correctss'].astype(object)
    Original_RNA_Data['color_matrix_seq_exprtss'] = Original_RNA_Data['color_matrix_seq_exprtss'].astype(object)

    Original_RNA_Data['nt_localized_info_matrix_seq_correctss'] = Original_RNA_Data[
        'nt_localized_info_matrix_seq_correctss'].astype(object)
    Original_RNA_Data['nt_localized_info_matrix_seq_exprtss'] = Original_RNA_Data[
        'nt_localized_info_matrix_seq_exprtss'].astype(object)

    Original_RNA_Data['embedding_generated'] = Original_RNA_Data['embedding_generated'].astype(object)
    Original_RNA_Data['embedding'] = Original_RNA_Data['embedding'].astype(object)
    Original_RNA_Data['RNA_ss'] = Original_RNA_Data['RNA_ss'].astype(object)

    # 2.1 Obtain the number of rows and columns of the data frame
    num_row_Original_RNA_Data = Original_RNA_Data.shape[0]
    num_col_Original_RNA_Data = Original_RNA_Data.shape[1]

    def load_and_pad_embedding(rna_name, seq, folder, max_length=410):
        embedding_path = f"{folder}/{rna_name}.npy"
        embedding = np.load(embedding_path)
        L, D = embedding.shape
        onehot = encode_onehot(seq)
        L_onehot, D_onehot = onehot.shape

        # 检查是否需要补零
        if L < max_length:
            padded_embedding = np.pad(embedding, ((0, max_length - L), (0, 0)), mode='constant')
            padded_onehot = np.pad(onehot, ((0, max_length - L_onehot), (0, 0)), mode='constant')
        else:
            padded_embedding = embedding[:max_length, :]
            padded_onehot = onehot[:max_length, :]

        # 拼接嵌入和One-hot编码
        combined_embedding = np.concatenate((padded_embedding, padded_onehot), axis=1)

        return combined_embedding

    # 2.2 Genarate a grayscale color matrix and nt localized info matrix for each pair of RNA Sequence and RNA Secondary Structure
    for i in range(num_row_Original_RNA_Data):
        Original_RNA_Data.at[i, 'RNA_ss'] = create_fixed_adjacency_matrix(
            Original_RNA_Data.at[i, 'RNA_secondary_structure'])
        Original_RNA_Data.at[i, 'embedding'] = load_and_pad_embedding(Original_RNA_Data.at[i, 'RNA_name'],
                                                                      Original_RNA_Data.at[i, 'RNA_seq_upper'],
                                                                      embedding_folder)
        Original_RNA_Data.at[i, 'embedding_generated'] = load_and_pad_embedding(Original_RNA_Data.at[i, 'RNA_name'],
                                                                                Original_RNA_Data.at[
                                                                                    i, 'RNA_seq_upper'],
                                                                                embedding_generated_folder)
        Original_RNA_Data.at[i, 'color_matrix_seq_correctss'] = RNA_matrix_3d_generator_canonical_bp(
            Original_RNA_Data.at[i, 'RNA_seq_upper'], Original_RNA_Data.at[i, 'RNA_secondary_structure'])
        Original_RNA_Data.at[i, 'color_matrix_seq_exprtss'] = RNA_matrix_3d_generator_canonical_bp(
            Original_RNA_Data.at[i, 'RNA_generated_seq'], Original_RNA_Data.at[i, 'RNA_secondary_structure'])
        Original_RNA_Data.at[i, 'nt_localized_info_matrix_seq_correctss'] = nt_localized_info_mat_generator(
            Original_RNA_Data.at[i, 'RNA_seq_upper'], Original_RNA_Data.at[i, 'RNA_secondary_structure'])
        Original_RNA_Data.at[i, 'nt_localized_info_matrix_seq_exprtss'] = nt_localized_info_mat_generator(
            Original_RNA_Data.at[i, 'RNA_generated_seq'], Original_RNA_Data.at[i, 'RNA_secondary_structure'])

    # 2.3 Normalize the MCC
    # Original_RNA_Data['Normalized_MCC'] = (Original_RNA_Data['MCC']+1)/2

    # For each RNA sample, the code generates two grayscale color matrix and two nt localized info matrix. One grayscale color matrix and one nt localized info matrix are from actual RNA secondary structure. Another
    # one grayscale color matrix and another one nt localized info matrix are from predicted RNA secondary structure.

    # 2.3.1.1 Obtain the grayscale color matrix and nt localized info matrix from the actual RNA secondary structure
    color_mat_3d_actual_RNA_structure = Original_RNA_Data[
        ['color_matrix_seq_correctss', 'nt_localized_info_matrix_seq_correctss', 'embedding', 'RNA_ss']].copy(deep=True)

    # 2.3.1.2 Add one column normalized MCC score for the corresponding grayscale color matrix
    color_mat_3d_actual_RNA_structure['dist_positive_negative_samples'] = np.nan
    color_mat_3d_actual_RNA_structure['dist_positive_negative_samples'] = 0

    # 2.3.2.1 Obtain the grayscale color matrix and nt localized info matrix from the predicted RNA secondary structure
    color_mat_3d_predicted_RNA_structure = Original_RNA_Data[
        ['color_matrix_seq_exprtss', 'nt_localized_info_matrix_seq_exprtss', 'dist_positive_negative_samples',
         'embedding_generated', 'RNA_ss']].copy(deep=True)

    # 2.3.2.2 Keep the grayscale color matrix whose corresponding normalized MCC is less than 1
    color_mat_3d_predicted_RNA_structure = color_mat_3d_predicted_RNA_structure[
        color_mat_3d_predicted_RNA_structure['dist_positive_negative_samples'] > 0]

    # 2.3.3 Rename the corresponding column name
    color_mat_3d_actual_RNA_structure = color_mat_3d_actual_RNA_structure.rename(
        columns={"color_matrix_seq_correctss": "color_matrix_utilized",
                 "nt_localized_info_matrix_seq_correctss": "nt_localized_info_matrix_utilized"})

    color_mat_3d_predicted_RNA_structure = color_mat_3d_predicted_RNA_structure.rename(
        columns={"color_matrix_seq_exprtss": "color_matrix_utilized",
                 "nt_localized_info_matrix_seq_exprtss": "nt_localized_info_matrix_utilized",
                 "embedding_generated": "embedding"})

    # 2.3.4 Combine two data sets
    color_matrix_data_set_combined = pd.concat(
        [color_mat_3d_actual_RNA_structure, color_mat_3d_predicted_RNA_structure], axis=0, ignore_index=True)

    # 2.5 Create the label column: 1 represents the positive sample and 0 represents the negative sample
    color_matrix_data_set_combined['RNA_Label'] = np.where(
        color_matrix_data_set_combined['dist_positive_negative_samples'] > 0, 0, 1)

    # 2.6 Obtain the number of rows in the color_matrix_data_set_combined
    num_row_color_matrix_data_set_combined = color_matrix_data_set_combined.shape[0]

    # 3.1 Obatin the size of each grayscale color matrix
    color_matrix_data_set_combined['color_mat_size'] = np.nan

    for i in range(num_row_color_matrix_data_set_combined):
        color_matrix_data_set_combined.at[i, 'color_mat_size'] = color_matrix_data_set_combined.iloc[i, 0].shape[2]

    # Obtain the maximum size of all grayscale color matrix and do the padding for all grayscale color matrix and all nt localized info matrix
    size_maximum = padding_length

    print("The maximum length of the RNA sequences is {}.".format(size_maximum))

    color_matrix_data_set_combined['color_matrix_utilized_for_CNN_padding'] = np.nan
    color_matrix_data_set_combined['color_matrix_utilized_for_CNN_padding'] = color_matrix_data_set_combined[
        'color_matrix_utilized_for_CNN_padding'].astype(object)
    color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'] = np.nan
    color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'] = \
    color_matrix_data_set_combined['nt_localized_info_matrix_utilized_for_CNN_padding'].astype(object)

    for i in range(num_row_color_matrix_data_set_combined):
        size_diff = size_maximum - color_matrix_data_set_combined.at[i, 'color_mat_size']
        size_diff_utilized = int(size_diff)
        color_matrix_data_set_combined.at[i, 'nt_localized_info_matrix_utilized_for_CNN_padding'] = np.pad(
            color_matrix_data_set_combined.iloc[i, 1], ((0, size_diff_utilized), (0, 0)), 'constant')

        if (size_diff % 2) == 0:
            num_padding_left_above = int(size_diff / 2)
            num_padding_right_below = int(size_diff / 2)
        else:
            num_padding_left_above = int(size_diff / 2)
            num_padding_right_below = int(size_diff / 2) + 1

        color_matrix_data_set_combined.at[i, 'color_matrix_utilized_for_CNN_padding'] = np.pad(
            color_matrix_data_set_combined.iloc[i, 0], ((0, 0), (num_padding_left_above, num_padding_right_below),
                                                        (num_padding_left_above, num_padding_right_below)), 'constant')

    # 3.2 Test the size of the grayscale color matrix after do the padding
    color_matrix_data_set_combined['color_mat_size_test_af_pad'] = np.nan
    color_matrix_data_set_combined['nt_localized_info_mat_size_test_af_pad'] = np.nan

    for i in range(num_row_color_matrix_data_set_combined):
        color_matrix_data_set_combined.at[i, 'color_mat_size_test_af_pad'] = \
        color_matrix_data_set_combined.at[i, 'color_matrix_utilized_for_CNN_padding'].shape[2]
        color_matrix_data_set_combined.at[i, 'nt_localized_info_mat_size_test_af_pad'] = \
        color_matrix_data_set_combined.at[i, 'nt_localized_info_matrix_utilized_for_CNN_padding'].shape[0]

    # 3.3 Shuffle the data
    color_matrix_data_set_combined = color_matrix_data_set_combined.sample(frac=1, random_state=59).reset_index(
        drop=True)

    # 4.1 Obtain the grayscale color matrix and label for each RNA samples. And transfer the df to np
    x = color_matrix_data_set_combined[
        ['color_matrix_utilized_for_CNN_padding', 'nt_localized_info_matrix_utilized_for_CNN_padding', 'embedding',
         'RNA_ss']]
    y = color_matrix_data_set_combined[['RNA_Label']]

    train_x_color_mat = x.loc[:, 'color_matrix_utilized_for_CNN_padding'].to_numpy()

    train_x_nt_localized_info_mat = x.loc[:, 'nt_localized_info_matrix_utilized_for_CNN_padding'].to_numpy()

    train_x_embedding = x.loc[:, 'embedding'].to_numpy()

    train_x_ss = x.loc[:, 'RNA_ss'].to_numpy()

    train_y = y.iloc[:, 0].to_numpy()

    # 4.2 Transfer the 1-D np array to 3-D np array
    train_x_color_mat_stack = train_x_color_mat[0][np.newaxis, :, :, :]

    for j, mat in enumerate(train_x_color_mat[1:]):
        color_mat_augmented = mat[np.newaxis, :, :, :]
        train_x_color_mat_stack = np.vstack((train_x_color_mat_stack, color_mat_augmented))

    train_x_nt_localized_info_mat_stack = train_x_nt_localized_info_mat[0][np.newaxis, np.newaxis, :, :]

    for j, mat in enumerate(train_x_nt_localized_info_mat[1:]):
        nt_localized_info_mat_augmented = mat[np.newaxis, np.newaxis, :, :]
        train_x_nt_localized_info_mat_stack = np.vstack(
            (train_x_nt_localized_info_mat_stack, nt_localized_info_mat_augmented))

    train_x_embedding_stack = train_x_embedding[0][np.newaxis, :, :]  # 增加一个新的维度，使其符合需要的形状

    for j, mat in enumerate(train_x_embedding[1:]):
        embedding_mat_augmented = mat[np.newaxis, :, :]  # 给每个矩阵增加一个新的维度
        train_x_embedding_stack = np.vstack((train_x_embedding_stack, embedding_mat_augmented))  # 合并到栈中

    # 处理 train_x_ss
    train_x_ss_stack = train_x_ss[0][np.newaxis, :, :, :]  # 增加一个新的维度，使其符合需要的形状

    for j, mat in enumerate(train_x_ss[1:]):
        ss_mat_augmented = mat[np.newaxis, :, :, :]  # 给每个矩阵增加一个新的维度
        train_x_ss_stack = np.vstack((train_x_ss_stack, ss_mat_augmented))  # 合并到栈中

    return train_x_color_mat_stack, train_x_nt_localized_info_mat_stack, train_y, train_x_embedding_stack, train_x_ss_stack




















