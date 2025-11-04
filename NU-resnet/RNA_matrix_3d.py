import numpy as np
import pandas as pd


def parse_vienna_to_pairs(vienna_structure):
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


########## Define the funtion to generate the color matrix for one pair of RNA Sequence and RNA Secondary Structure

def RNA_matrix_3d_generator(RNA_seq, RNA_ss):
    #  2.1 obtain the index for the paired bases 
    parse_vienna_to_pairs(RNA_ss)
    pairs_index, pairs_pk_index = parse_vienna_to_pairs(RNA_ss)
    
    # 2.2 construct the colormap matrix 
    RNA_seq_split = list(RNA_seq)
    grayscale_mat_c_1 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_2 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_3 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_4 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    
    # 2.3 assign value to each element of the colormap matrix
    row_name_colorm = list(grayscale_mat_c_1.index)
    col_name_colorm = list(grayscale_mat_c_1.columns)
    
    # 3.1 In the diagonal, assign vector to each nucleotide
    for i, row_base in enumerate(row_name_colorm, start=1):
        if row_base == 'A':
            grayscale_mat_c_1.iloc[i-1, i-1] = 1
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'U':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 1
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'C':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 1
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'G':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 1
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
    

##########

def RNA_matrix_3d_generator_canonical_bp(RNA_seq, RNA_ss):
    #  2.1 obtain the index for the paired bases 
    parse_vienna_to_pairs(RNA_ss)
    pairs_index, pairs_pk_index = parse_vienna_to_pairs(RNA_ss)
    
    # 2.2 construct the colormap matrix 
    RNA_seq_split = list(RNA_seq)
    grayscale_mat_c_1 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_2 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_3 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    grayscale_mat_c_4 = pd.DataFrame(index = RNA_seq_split, columns = RNA_seq_split)
    
    # 2.3 assign value to each element of the colormap matrix
    row_name_colorm = list(grayscale_mat_c_1.index)
    col_name_colorm = list(grayscale_mat_c_1.columns)
    
    # 3.1 In the diagonal, assign vector to each nucleotide
    for i, row_base in enumerate(row_name_colorm, start=1):
        if row_base == 'A':
            grayscale_mat_c_1.iloc[i-1, i-1] = 1
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'U':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 1
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'C':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 1
            grayscale_mat_c_4.iloc[i-1, i-1] = 0
        elif row_base == 'G':
            grayscale_mat_c_1.iloc[i-1, i-1] = 0
            grayscale_mat_c_2.iloc[i-1, i-1] = 0
            grayscale_mat_c_3.iloc[i-1, i-1] = 0
            grayscale_mat_c_4.iloc[i-1, i-1] = 1
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


##########