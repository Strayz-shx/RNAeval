import os
import math
import random
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from RNAseqdataset import encode_onehot
from model import RNAInception, BasicBlock

# ============================================================
# 1. MCC计算和文件过滤工具函数
# ============================================================

def calculate_average(tp, tn, fp, fn):
    """根据 TP/TN/FP/FN 计算各项指标"""
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    MC = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if MC == 0:
        MCC = 0
    else:
        MCC = ((tp * tn) - (fp * fn)) / MC

    if (precision + sensitivity) == 0:
        F1 = 0
    else:
        F1 = 2 * precision * sensitivity / (precision + sensitivity)
    acc = (tp + tn) / (tp + fn + fp + tn)
    return precision, sensitivity, MCC, F1, acc


def evaluate_similarity(path, path1):
    """比较两个 bpseq 文件的第三列（配对索引），返回各项指标"""
    nodes1, nodes2 = [], []
    tp = tn = fp = fn = 0

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                nodes1.append(parts[2])

    with open(path1, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                nodes2.append(parts[2])

    length = min(len(nodes1), len(nodes2))
    for i in range(length):
        v1, v2 = int(nodes1[i]), int(nodes2[i])
        if v1 == v2 and v1 != 0:
            tp += 1
        elif v1 == v2 and v1 == 0:
            tn += 1
        elif v1 != v2 and v1 != 0 and v2 == 0:
            fp += 1
        else:
            fn += 1

    return calculate_average(tp, tn, fp, fn)


def filter_mcc_one_files(neg_dir, pos_dir):
    """
    过滤掉与正样本MCC=1的负样本文件
    返回：该文件夹下MCC不为1的文件名列表
    """
    neg_files = set(f for f in os.listdir(neg_dir) if f.endswith('.bpseq'))
    pos_files = set(f for f in os.listdir(pos_dir) if f.endswith('.bpseq'))
    common_files = neg_files & pos_files

    valid_files = []
    for filename in common_files:
        neg_path = os.path.join(neg_dir, filename)
        pos_path = os.path.join(pos_dir, filename)
        try:
            _, _, mcc, _, _ = evaluate_similarity(neg_path, pos_path)
            if abs(mcc - 1.0) >= 1e-10:
                valid_files.append(filename)
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    return valid_files


# ============================================================
# 2. 自定义Dataset
# ============================================================

class CrossValRNASeqDataset(Dataset):
    """
    五折交叉验证专用Dataset
    """
    def __init__(self, pos_dir, neg_dir, rna_fm_dir, file_list):
        """
        Args:
            pos_dir: 正样本bpseq文件夹
            neg_dir: 负样本bpseq文件夹
            rna_fm_dir: RNA-FM特征npy文件夹
            file_list: [(filename, label), ...]  label: 1=正样本, 0=负样本
        """
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.rna_fm_dir = rna_fm_dir
        self.samples = []
        self.skipped_files = []

        for fname, label in file_list:
            if label == 1:
                bpseq_path = os.path.join(pos_dir, fname)
            else:
                bpseq_path = os.path.join(neg_dir, fname)

            npy_path = os.path.join(rna_fm_dir, fname.replace('.bpseq', '.npy'))
            if not os.path.exists(bpseq_path) or not os.path.exists(npy_path):
                self.skipped_files.append((fname, "missing_file"))
                continue

            # 检查长度一致性：bpseq行数应该等于npy的第一维
            try:
                bpseq_len = self._get_bpseq_len(bpseq_path)
                npy_shape = np.load(npy_path).shape
                if bpseq_len != npy_shape[0]:
                    self.skipped_files.append((fname, f"length_mismatch(bpseq={bpseq_len},npy={npy_shape[0]})"))
                    continue
            except Exception as e:
                self.skipped_files.append((fname, f"check_error:{e}"))
                continue

            self.samples.append((bpseq_path, npy_path, label))

        if len(self.skipped_files) > 0:
            print(f"  警告: 跳过 {len(self.skipped_files)} 个不一致/缺失的样本")
            # 只打印前5个
            for fname, reason in self.skipped_files[:5]:
                print(f"    {fname}: {reason}")
            if len(self.skipped_files) > 5:
                print(f"    ... 还有 {len(self.skipped_files)-5} 个")

    def _get_bpseq_len(self, bpseq_path):
        """获取bpseq文件的序列长度（行数）"""
        count = 0
        with open(bpseq_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_len = 734
        bpseq_path, npy_path, label = self.samples[idx]

        # 读取bpseq数据
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

        flatten_x = torch.zeros((max_len, 645))
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
                parts = line.strip().split()
                if len(parts) >= 3:
                    index = int(parts[0]) - 1
                    base = parts[1]
                    pair_index = int(parts[2]) - 1
                    sequences.append(base)
                    if index != -1 and pair_index != -1:
                        list1.append(index)
                        list2.append(pair_index)
            edge_matrix = np.array([list1, list2])

        return sequences, edge_matrix


# ============================================================
# 3. 训练/验证函数（RNAeval模型）
# ============================================================

def train_rivas(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in train_loader:
        struc, x, labels = batch
        x, struc, labels = x.float().to(device), struc.float().to(device), labels.float().to(device)
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(struc, x)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    return avg_loss, accuracy


def valid_rivas(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            struc, x, labels = batch
            x, struc, labels = x.float().to(device), struc.float().to(device), labels.float().to(device)
            labels = labels.long()

            outputs = model(struc, x)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    return avg_loss, accuracy


# ============================================================
# 4. Logger工具
# ============================================================

class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log
        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger


def make_logger(out_dir, fold_idx=0):
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(out_dir, f"fold_{fold_idx}_{time_str}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir


# ============================================================
# 5. 模型构建函数
# ============================================================

def build_model(config, device):
    if config['model'] == 0:
        from model import ResNet_18_pair_grayscale_mat_nt_localized_info_mat
        model = ResNet_18_pair_grayscale_mat_nt_localized_info_mat().to(device)
        model_name = "numo_resnet"
    elif config['model'] == 1:
        model = RNAInception(out_channels=512).to(device)
        model_name = "RNAInception"
    elif config['model'] == 2:
        from model import RNAInception_modify
        model = RNAInception_modify(config["out_channels"]).to(device)
        model_name = "Inception_modify"
    elif config['model'] == 3:
        from model import RNACnn_trans
        model = RNACnn_trans(BasicBlock, layers=[2,2,2,2], layers_struc=[2,2,2,2],
                             out_channels=config["out_channels"]).to(device)
        model_name = "resnet"
    elif config['model'] == 4:
        from model import ResNet_18_grayscale_mat
        model = ResNet_18_grayscale_mat().to(device)
        model_name = "nu_resnet"
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
    return model, model_name


# ============================================================
# 6. 五折交叉验证主函数
# ============================================================

def five_fold_cross_validation(config, pos_dir, neg_dirs, rna_fm_dir):
    """
    五折交叉验证主函数

    逻辑：
    1. 前4个文件夹（A,B,C,D）分别过滤MCC=1，各选 578/578/578/577 个负样本（互不相同）
    2. 第5个文件夹（E）过滤MCC=1后，全部使用（577个）
    3. 根据选中的负样本文件名，找对应的正样本
    4. 每折：4份(负样本+对应正样本) → 训练集，1份(负样本+对应正样本) → 验证集
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assert len(neg_dirs) == 5, f"需要提供5个负样本文件夹，当前提供了 {len(neg_dirs)} 个"

    # ==================== Step 1: 前4个文件夹过滤并选取指定数量 ====================
    print("\n========== Step 1: 前4个文件夹过滤MCC=1并选取负样本 ==========")

    fold_neg_files = []  # 每个fold的负样本文件名列表（仅文件名，不含路径）
    used_files = set()     # 已使用的文件名，确保前4个文件夹互不相同

    # 前4个文件夹：各选 578/578/578/577 个
    target_counts = [578, 578, 578, 577]

    for i in range(4):
        neg_dir = neg_dirs[i]
        target = target_counts[i]
        print(f"\n处理文件夹 {i+1} ({neg_dir}):")

        if not os.path.exists(neg_dir):
            print(f"  警告: 文件夹不存在，跳过")
            fold_neg_files.append([])
            continue

        # 过滤MCC=1
        valid_files = filter_mcc_one_files(neg_dir, pos_dir)
        print(f"  过滤后有效文件: {len(valid_files)}")

        # 排除已使用的文件
        available = [f for f in valid_files if f not in used_files]
        print(f"  排除已使用后可用: {len(available)} (需要 {target})")

        if len(available) < target:
            print(f"  警告: 可用文件不足，使用全部 {len(available)} 个")
            selected = available
        else:
            random.seed(42 + i)
            selected = random.sample(available, target)

        # 标记为已使用
        for f in selected:
            used_files.add(f)

        fold_neg_files.append(selected)
        print(f"  最终选取: {len(selected)} 个")

    # ==================== Step 2: 第5个文件夹选取577个 ====================
    print("\n========== Step 2: 第5个文件夹过滤并选取577个 ==========")
    neg_dir_e = neg_dirs[4]
    print(f"\n处理文件夹 5 ({neg_dir_e}):")

    if not os.path.exists(neg_dir_e):
        raise ValueError(f"第5个文件夹不存在: {neg_dir_e}")

    valid_files_e = filter_mcc_one_files(neg_dir_e, pos_dir)
    print(f"  过滤后有效文件: {len(valid_files_e)}")

    # 第5个文件夹选取577个（可以与前面重复，因为是不同方法生成的不同结构）
    target_e = 577
    if len(valid_files_e) < target_e:
        print(f"  警告: 有效文件不足，使用全部 {len(valid_files_e)} 个")
        selected_e = valid_files_e
    else:
        random.seed(42 + 4)
        selected_e = random.sample(valid_files_e, target_e)

    fold_neg_files.append(selected_e)
    print(f"  选取: {len(selected_e)} 个")

    # ==================== Step 3: 根据负样本找对应的正样本 ====================
    print("\n========== Step 3: 匹配正样本 ==========")

    pos_files_all = set(f for f in os.listdir(pos_dir) if f.endswith('.bpseq'))
    # 只保留有对应.npy的正样本
    pos_files_all = {f for f in pos_files_all
                     if os.path.exists(os.path.join(rna_fm_dir, f.replace('.bpseq', '.npy')))}

    fold_samples = []  # 每个fold: [(filename, label), ...]

    for i in range(5):
        neg_files = fold_neg_files[i]
        neg_dir = neg_dirs[i]

        # 找这些负样本对应的正样本（同名文件）
        matched_pos = [f for f in neg_files if f in pos_files_all]
        # 找有正样本对应的负样本（确保正负样本一一对应）
        matched_neg = [f for f in neg_files if f in pos_files_all]

        # 构建样本列表: 每个负样本对应一个正样本
        sample_list = []
        for fname in matched_neg:
            sample_list.append((fname, 0))  # 负样本
            sample_list.append((fname, 1))  # 对应的正样本

        fold_samples.append({
            'neg_dir': neg_dir,
            'neg_files': matched_neg,
            'pos_files': matched_pos,
            'sample_list': sample_list,
            'num_neg': len(matched_neg),
            'num_pos': len(matched_pos),
            'total': len(sample_list)
        })

        print(f"\nFold {i+1} ({os.path.basename(neg_dir)}):")
        print(f"  负样本: {len(matched_neg)} 个")
        print(f"  对应正样本: {len(matched_pos)} 个")
        print(f"  总样本数(neg+pos): {len(sample_list)} 个")

    # ==================== Step 4: 五折交叉验证 ====================
    print(f"\n========== Step 4: 五折交叉验证 ==========")

    res_dir = config.get("res_dir",
                          r"/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/RNAeval-main/cross_val_result")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    fold_results = []

    for fold in range(5):
        print(f"\n{'='*60}")
        print(f"  Fold {fold + 1} / 5")
        print(f"{'='*60}")

        # 验证集：当前fold的样本
        val_data = fold_samples[fold]
        val_neg_dir = val_data['neg_dir']
        val_sample_list = val_data['sample_list']

        # 训练集：其余4个fold的样本合并
        train_sample_list = []
        for j in range(5):
            if j != fold:
                train_sample_list.extend(fold_samples[j]['sample_list'])

        print(f"  训练集: {len(train_sample_list)} 个样本")
        print(f"  验证集: {len(val_sample_list)} 个样本 (来自 {os.path.basename(val_neg_dir)})")

        # 构建Dataset
        # 训练集：负样本来自各自的文件夹，正样本来自pos_dir
        # 由于训练集包含多个文件夹的负样本，需要特殊处理
        # 这里简化：每个fold的负样本只来自一个文件夹，所以训练集负样本来自4个不同文件夹
        # 但CrossValRNASeqDataset只支持一个neg_dir，我们需要构建一个组合Dataset

        train_dataset = _build_combined_dataset(pos_dir, neg_dirs, rna_fm_dir, fold, fold_samples, is_train=True)
        val_dataset = CrossValRNASeqDataset(
            pos_dir=pos_dir,
            neg_dir=val_neg_dir,
            rna_fm_dir=rna_fm_dir,
            file_list=val_sample_list
        )

        print(f"  训练集Dataset: {len(train_dataset)} 个")
        print(f"  验证集Dataset: {len(val_dataset)} 个")

        # DataLoader
        dataloader_train = DataLoader(train_dataset, batch_size=config["batch_size"],
                                      shuffle=True, num_workers=4, drop_last=False)
        dataloader_valid = DataLoader(val_dataset, batch_size=config["batch_size"],
                                      shuffle=False, num_workers=4, drop_last=False)

        # 模型
        model, model_name = build_model(config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                      weight_decay=config['weight_decay'])
        loss_fn = nn.CrossEntropyLoss()

        # Logger
        logger, log_dir = make_logger(res_dir, fold_idx=fold)
        logger.info(f"===== Fold {fold+1}/5 =====")
        logger.info(f"model_name: {model_name}, loss: CrossEntropyLoss, lr: {config['lr']}")
        logger.info(f"train_samples: {len(train_dataset)}, val_samples: {len(val_dataset)}")
        logger.info(f"val_from: {os.path.basename(val_neg_dir)}")

        best_acc = 0
        best_epoch = 0

        # 训练循环
        for epoch in range(1, config['num_of_epochs'] + 1):
            print(f"\n------ Fold {fold+1} Epoch {epoch}/{config['num_of_epochs']} ------")

            train_loss, train_acc = train_rivas(model, device, dataloader_train, optimizer, loss_fn)
            val_loss, val_acc = valid_rivas(model, device, dataloader_valid, loss_fn)

            logger.info(f"Epoch[{epoch:03d}/{config['num_of_epochs']:03d}] "
                        f"Train acc:{train_acc:.4f} Valid acc:{val_acc:.4f} "
                        f"Train loss:{train_loss:.4f} Valid loss:{val_loss:.4f}")

            print(f"Train acc: {train_acc:.4f}%, Valid acc: {val_acc:.4f}%")

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_name = f"checkpoint_fold{fold+1}_best.pth"
                best_model_path = os.path.join(log_dir, best_name)
                torch.save(model.state_dict(), best_model_path)
                print(f"------ 保存最佳模型 (acc={val_acc:.4f}) ------")

            # 最后一个epoch也保存
            if epoch == config['num_of_epochs']:
                final_name = f"checkpoint_fold{fold+1}_epoch{epoch}.pth"
                final_path = os.path.join(log_dir, final_name)
                torch.save(model.state_dict(), final_path)

        logger.info(f"Fold {fold+1} done, best_valid_acc: {best_acc:.4f} in epoch: {best_epoch}")
        fold_results.append({
            'fold': fold + 1,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'val_source': os.path.basename(val_neg_dir),
            'log_dir': log_dir
        })

        print(f"\nFold {fold+1} 完成，最佳验证准确率: {best_acc:.4f}% (Epoch {best_epoch})")

    # ==================== Step 5: 汇总结果 ====================
    print(f"\n{'='*60}")
    print("  五折交叉验证结果汇总")
    print(f"{'='*60}")

    total_best_acc = sum(r['best_acc'] for r in fold_results)
    avg_acc = total_best_acc / len(fold_results) if fold_results else 0

    logger_summary, summary_dir = make_logger(res_dir, fold_idx="summary")
    logger_summary.info("===== 五折交叉验证汇总 =====")

    for r in fold_results:
        info = (f"Fold {r['fold']}: Best Valid Acc = {r['best_acc']:.4f}% "
                f"(Epoch {r['best_epoch']}, ValSource={r['val_source']}, "
                f"Train={r['train_samples']}, Val={r['val_samples']})")
        print(info)
        logger_summary.info(info)

    summary_info = f"\n平均验证准确率: {avg_acc:.4f}%"
    print(summary_info)
    logger_summary.info(summary_info)

    # 保存汇总到文件
    summary_file = os.path.join(summary_dir, "cross_val_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("五折交叉验证结果汇总\n")
        f.write("="*60 + "\n")
        for r in fold_results:
            f.write(f"Fold {r['fold']}: Best Valid Acc = {r['best_acc']:.4f}% "
                    f"(Epoch {r['best_epoch']}, ValSource={r['val_source']})\n")
        f.write(f"\n平均验证准确率: {avg_acc:.4f}%\n")
    print(f"\n汇总结果已保存到: {summary_file}")

    return fold_results, avg_acc


def _build_combined_dataset(pos_dir, neg_dirs, rna_fm_dir, exclude_fold, fold_samples, is_train=True):
    """
    构建组合Dataset，训练集包含多个文件夹的负样本
    """
    all_samples = []
    for j in range(5):
        if j != exclude_fold:
            for fname, label in fold_samples[j]['sample_list']:
                if label == 1:
                    bpseq_path = os.path.join(pos_dir, fname)
                else:
                    bpseq_path = os.path.join(fold_samples[j]['neg_dir'], fname)
                npy_path = os.path.join(rna_fm_dir, fname.replace('.bpseq', '.npy'))
                if os.path.exists(bpseq_path) and os.path.exists(npy_path):
                    all_samples.append((bpseq_path, npy_path, label))

    # 创建简单的Dataset
    class SimpleDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    # 但我们需要保持和CrossValRNASeqDataset相同的输出格式
    # 所以直接返回一个使用相同逻辑的dataset
    # 这里我们构造file_list然后使用CrossValRNASeqDataset的变体

    # 实际上，由于训练集包含多个neg_dir，我们需要一个能处理多个neg_dir的Dataset
    # 简化方案：直接构造一个包含所有样本的Dataset

    class CombinedDataset(Dataset):
        def __init__(self, samples):
            # 过滤长度不一致的样本
            valid_samples = []
            skipped = 0
            for bpseq_path, npy_path, label in samples:
                valid_samples.append((bpseq_path, npy_path, label))
            self.samples = valid_samples
            if skipped > 0:
                print(f"  CombinedDataset 跳过 {skipped} 个长度不一致样本")
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            bpseq_path, npy_path, label = self.samples[idx]
            max_len = 734

            # 读取bpseq数据
            seq, edge = _parse_bpseq(bpseq_path)
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

            flatten_x = torch.zeros((max_len, 645))
            flatten_x[:x_features.shape[0], :x_features.shape[1]] = x_features
            flatten_x = flatten_x.double()

            return adj_matrix, flatten_x, label

    return CombinedDataset(all_samples)


def _parse_bpseq(bpseq_path):
    """解析bpseq文件"""
    sequences = []
    list1 = []
    list2 = []

    with open(bpseq_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                index = int(parts[0]) - 1
                base = parts[1]
                pair_index = int(parts[2]) - 1
                sequences.append(base)
                if index != -1 and pair_index != -1:
                    list1.append(index)
                    list2.append(pair_index)
        edge_matrix = np.array([list1, list2])

    return sequences, edge_matrix


# ============================================================
# 7. 参数解析
# ============================================================

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model", type=int, default=1,
                        help='0:Numo_resnet; 1:CNN_inception; 2:Inception_modify; 3:CNN_TRANS; 4:NU-resnet')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pos_dir", type=str,
                        default="/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/RE-datasetA/TrainSetA",
                        help="正样本bpseq文件夹路径")
    parser.add_argument("--neg_dirs", type=str, nargs='+',
                        default=[
                            "/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/RE-datasetA/contrafold_train",
                            "/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/RE-datasetA/mxfold2_train",
                            "/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/RE-datasetA/bpfold_train",
                            "/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/RE-datasetA/rnafm_train",
                            "/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/RE-datasetA/pre_Atrainset",
                        ],
                        help="5个负样本bpseq文件夹路径（空格分隔）")
    parser.add_argument("--rna_fm_dir", type=str,
                        default="/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/data/fm_representations/TrainA",
                        help="RNA-FM特征npy文件夹路径")
    parser.add_argument("--res_dir", type=str,
                        default="/home/bingxing2/home/scx7auf/Shihaoxuan/graduate/RNAeval-main/cross_val_result",
                        help="结果保存路径")

    gcn_config = {
        "cnn_channels": 256,
        "features": 645,
        "hidden_channels": 32,
        "out_channels": 512,
    }

    # 兼容 Jupyter 环境，过滤掉 -f 参数
    import sys
    argv = sys.argv
    if '-f' in argv:
        idx = argv.index('-f')
        argv = argv[:idx]  # 移除 -f 及其后面的参数

    args = parser.parse_args(argv[1:])
    training_config = vars(args)
    training_config.update(gcn_config)
    return training_config


    
# ============================================================
# 8. 主入口
# ============================================================

if __name__ == '__main__':
    config = get_training_args()

    neg_dirs = config['neg_dirs']
    if len(neg_dirs) != 5:
        raise ValueError(f"需要提供5个负样本文件夹，当前提供了 {len(neg_dirs)} 个: {neg_dirs}")

    print("="*60)
    print("  五折交叉验证配置")
    print("="*60)
    print(f"  正样本: {config['pos_dir']}")
    print(f"  负样本文件夹:")
    for i, d in enumerate(neg_dirs):
        print(f"    [{i+1}] {d}")
    print(f"  RNA-FM: {config['rna_fm_dir']}")
    print(f"  模型类型: {config['model']}")
    print(f"  Epochs: {config['num_of_epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  结果保存: {config['res_dir']}")
    print("="*60)

    results, avg_acc = five_fold_cross_validation(
        config=config,
        pos_dir=config['pos_dir'],
        neg_dirs=neg_dirs,
        rna_fm_dir=config['rna_fm_dir']
    )

    print(f"\n{'='*60}")
    print(f"  五折交叉验证完成！")
    print(f"  平均验证准确率: {avg_acc:.4f}%")
    print(f"{'='*60}")