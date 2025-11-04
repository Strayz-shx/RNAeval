import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import RNASeqDataset,Nu_res_Dataset,RNA_family_Dataset, RNA_family_Dataset_for_nu, RNA_ent_Dataset, RNA_ent_Dataset_for_nu
from model import RNAInception, Cnn_trans, ResNet_18_grayscale_mat, RNACnn_trans, BasicBlock, RNAInception_modify, ResNet_18_pair_grayscale_mat_nt_localized_info_mat
from torch.utils.data import random_split


def collate_fn(batch):
    max_len = max(data.x.shape[0] for data in batch)   # 取整个数据集的最大长度

    adj_matrics = []
    flattened_features = []
    labels = []

    for data in batch:
        x = data.x
        edge_index = data.edge_index
        y = data.y

        adj_matrix = torch.zeros((max_len, max_len))
        adj_matrix[edge_index[0], edge_index[1]] = 1

        dot_product_matrix = torch.mm(x, x.t())

        padded_dot_product_matrix = torch.zeros((max_len, max_len))
        padded_dot_product_matrix[:dot_product_matrix.shape[0], :dot_product_matrix.shape[1]] = dot_product_matrix

        combined_matrix = torch.stack([adj_matrix, padded_dot_product_matrix], dim=-1)

        flatten_x = torch.zeros((max_len, 645))  # 补齐data.x
        flatten_x[:x.shape[0], :x.shape[1]] = x

        adj_matrics.append(combined_matrix)
        labels.append(y)
        flattened_features.append(flatten_x)

    adj_matrics = torch.stack(adj_matrics)
    labels = torch.stack(labels)
    flattened_features = torch.stack(flattened_features)

    return adj_matrics, labels, flattened_features


def train(model, device, train_loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in train_loader:
        color_mat, nt_localized_mat, label = batch
        color_mat, nt_localized_mat, label = color_mat.to(device), nt_localized_mat.to(device), label.to(device)

        label = label.long()
        optimizer.zero_grad()

        outputs = model(color_mat)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == label).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += label.size(0)
        num_batches += 1
    # 计算平均指标
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples * 100  # 百分比形式

    return avg_loss, accuracy


def valid(model, device, val_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            color_mat, nt_localized_mat, label = batch
            color_mat, nt_localized_mat, label = color_mat.to(device), nt_localized_mat.to(device), label.to(device)  # nu-resnet只使用color_mat

            label = label.long()

            outputs = model(color_mat)
            loss = criterion(outputs, label)

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == label).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += label.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy


def train_numo(model, device, train_loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in train_loader:
        color_mat, nt_localized_mat, label = batch
        color_mat, nt_localized_mat, label = color_mat.to(device), nt_localized_mat.to(device), label.to(device)  # numo-resnet模型使用color_mat和nt_localized_mat

        label = label.long()
        optimizer.zero_grad()

        outputs = model(color_mat, nt_localized_mat)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == label).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += label.size(0)
        num_batches += 1
    # 计算平均指标
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples * 100  # 百分比形式

    return avg_loss, accuracy


def valid_numo(model, device, val_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            color_mat, nt_localized_mat, label = batch
            color_mat, nt_localized_mat, label = color_mat.to(device), nt_localized_mat.to(device), label.to(device)

            label = label.long()

            outputs = model(color_mat, nt_localized_mat)
            loss = criterion(outputs, label)

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == label).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += label.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy


def train_rivas(model, device, train_loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in train_loader:
        struc, x, labels = batch

        # 数据转换和设备迁移
        # Move data to device (GPU/CPU)
        x, struc, labels = x.float().to(device), struc.float().to(device), labels.float().to(
            device)

        labels = labels.long()

        optimizer.zero_grad()

        # 前向传播
        outputs = model(struc, x)

        loss = criterion(outputs, labels)  # outputs需要是[B, 2]形状

        # 反向传播
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)  # 获取预测类别
        correct = (predicted == labels).sum().item()

        # 统计指标
        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)
        num_batches += 1

    # 计算平均指标
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples * 100  # 百分比形式

    return avg_loss, accuracy


def valid_rivas(model, device, val_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            struc,x, labels = batch
            # Move data to device (GPU/CPU)
            x, struc, labels = x.float().to(device), struc.float().to(device), labels.float().to(
                device)

            labels = labels.long()

            outputs = model(struc, x)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy


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

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir):
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建logger
    path_log = os.path.join(log_dir, "log.log")  # log的名字
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir

def main_eval_rna(config, dataset_train):
    writer = SummaryWriter(log_dir='runs/mock_accuracy')
    # ------------------------------------ step 1/4 : 定义数据------------------------------------
    # 数据集总大小
    dataset_size = len(dataset_train)

    # 计算训练集和验证集的大小(8:2)
    train_size = int(config["trainset_size"] * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(dataset_train, [train_size, val_size], generator=generator)

    dataloader_train = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    dataloader_valid = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # ------------------------------------ step 2/4 : 定义网络------------------------------------

    if config['model'] == 0:
        model = ResNet_18_pair_grayscale_mat_nt_localized_info_mat().to(device)
        # model = torch.nn.DataParallel(model, device_ids=[2, 3])  #
        # model.to(device)
        model_name = "numo_resnet"

    if config['model'] == 1:
        model = RNAInception(out_channels=512).to(device)
        # model = torch.nn.DataParallel(model, device_ids=[2, 3])
        # model.to(device)
        model_name = "RNAInception"

    if config['model'] == 2:

        model = RNAInception_modify(config["out_channels"]).to(device)
        # model = torch.nn.DataParallel(model, device_ids=[2, 3])
        # model.to(device)
        model_name = "Inception_modify"

    if config['model'] == 3:
        model = RNACnn_trans(BasicBlock, layers=[2,2,2,2], layers_struc=[2,2,2,2], out_channels=config["out_channels"]).to(device)
        # model = RNACnn_trans(Bottleneck, layers=[3, 4, 23, 3], layers_struc=[3, 4, 23, 3],
        #                      out_channels=config["out_channels"]).to(device)
        # model = torch.nn.DataParallel(model, device_ids=[2,3])
        # model.to(device)
        model_name = "resnet"

    if config['model'] == 4:
        model = ResNet_18_grayscale_mat().to(device)
        # model = torch.nn.DataParallel(model, device_ids=[2,3])
        # model.to(device)
        model_name = "nu_resnet"

    # ------------------------------------ step 3/4 : 定义损失函数和优化器 ------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    loss_fn = nn.CrossEntropyLoss()

    res_dir = r"model_result"  # 模型保存路径
    logger, log_dir = make_logger(res_dir)
    best_test_loss = float('inf')  # 初始化为一个很大的值
    best_epoch = 0  # 保存最佳epoch的编号
    best_acc = 0

    logger.info("本次训练的备注信息：")
    logger.info("model_name:{}\nloss_function:{}\nlearning_rate:{}\n".format(model_name, loss_fn, config["lr"]))

    for epoch in range(1, config['num_of_epochs']):
        print("------第 {} 轮训练开始------".format(epoch))
        if config["dataset"] == 1:
            avg_loss, train_accuracy = train_rivas(model, device, train_loader=dataloader_train, optimizer=optimizer,
                                             criterion=loss_fn)
            avg_loss, valid_accuracy = valid_rivas(model, device, val_loader=dataloader_valid, criterion=loss_fn)
            train_loss, train_mse, train_rmse, test_loss, test_mse, test_rmse = 0., 0., 0., 0., 0., 0.

        elif config['dataset'] == 2:
            avg_loss, train_accuracy = train(model, device, train_loader=dataloader_train, optimizer=optimizer,
                                             criterion=loss_fn)
            avg_loss, valid_accuracy = valid(model, device, val_loader=dataloader_valid, criterion=loss_fn)
            train_loss, train_mse, train_rmse, test_loss, test_mse, test_rmse = 0., 0., 0., 0., 0., 0.

        elif config['dataset'] == 3:
            avg_loss, train_accuracy = train_numo(model, device, train_loader=dataloader_train, optimizer=optimizer,
                                             criterion=loss_fn)
            avg_loss, valid_accuracy = valid_numo(model, device, val_loader=dataloader_valid, criterion=loss_fn)
            train_loss, train_mse, train_rmse, test_loss, test_mse, test_rmse = 0., 0., 0., 0., 0., 0.

        # scheduler.step(test_loss)

        logger.info("Epoch[{:0>3}/{:0>3}]\n"
                    "Train acc:{:.4f} Valid acc:{:.4f}\n"
                    "Train loss:{:.4f} Train mse:{:.4f}\n"
                    "Valid loss:{:.4f} Valid mse:{:.4f}\n"
                    "Train rmse:{:.4f} Valid rmse:{:.4f}\n"
                    .format(epoch, config['num_of_epochs'],train_accuracy, valid_accuracy, train_loss, train_mse, test_loss, test_mse, train_rmse,
                            test_rmse))

        # 分类模型保存结果
        if valid_accuracy > best_acc or epoch==config['num_of_epochs']:
            best_acc = valid_accuracy
            best_name = "checkpoint_{}.pth".format(epoch) if epoch == config['num_of_epochs'] else "checkpoint_best.pth"
            best_epoch = epoch
            best_model_path = os.path.join(log_dir, best_name)
            torch.save(model.state_dict(), best_model_path)
            print("------保存当前最佳模型------")

        # if test_loss < best_test_loss or epoch == config['num_of_epochs']:
        #     best_test_loss = test_loss
        #     best_name = "checkpoint_{}.pth".format(epoch) if epoch == config['num_of_epochs'] else "checkpoint_best.pth"
        #     best_epoch = epoch
        #     best_model_path = os.path.join(log_dir, best_name)
        #     torch.save(model.state_dict(), best_model_path)
        #     print("------保存当前最佳模型------")

        writer.add_scalar(tag="train_loss",  # 可以暂时理解为图像的名字
                          scalar_value=train_loss,  # 纵坐标的值
                          global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        writer.add_scalar(tag="test_loss",
                          scalar_value=test_loss,
                          global_step=epoch
                          )

    logger.info("{} done, best_test_loss: {:.4f} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_test_loss, best_epoch))


def get_training_args():
    parser = argparse.ArgumentParser()

    # 训练相关
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=200)
    parser.add_argument("--lr", type=float, help="model learning rate", default=0.0001)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=1e-4)
    parser.add_argument("--model", type=int, help='0:Numo_resnet;1:CNN_inception;2:消融实验 CNN_inception;3:CNN_TRANS;4:NU-resnet', default=1)
    parser.add_argument("--batch_size", type=int, help='批数量', default=8)
    parser.add_argument("--dataset", type=int, help='数据库类型,1:用于RNAeval模型,2:用于nu_resnet模型,3:用于numo_resnet模型', default=1)
    parser.add_argument("--trainset_size", type=float, help="训练集比例", default=0.8)

    gcn_config = {
        "cnn_channels": 256,
        "features": 645,   # 初始节点特征
        "hidden_channels": 32,   # 隐藏层维度
        "out_channels": 512,  # resnet18/34 512  resnet50/101 2048
    }

    args = parser.parse_args()
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    training_config.update(gcn_config)

    return training_config



if __name__ == '__main__':
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:2")  # 训练代码所用的设备
    """
    RNASeqDataset用于预处理RNAeval模型的数据
    可用于结构判别任务（非跨家族）
    """
    dataset = RNASeqDataset(bpseq_pos_dir=r"data/RE-datasetA/TrainSetA", bpseq_neg_dir=r"data/RE-datasetA/pre_Atrainset",
                            rna_fm_dir="/home/Shihx/RNA-FM/redevelop/TrainA/representations")

    """
    Nu_res_Dataset用于预处理可用于Nu-resnet和NUMO-resnet两个模型的数据
    可用于结构判别任务（非跨家族）
    """
    # dataset = Nu_res_Dataset(bpseq_pos_dir=r"data/RE-datasetA/TrainSetA", bpseq_neg_dir=r"data/RE-datasetA/pre_Atrainset")

    """
    RNA_family_Dataset用于leave-one-family-out实验
    每次去除一个家族，该家族作为测试集
    RNA_family_Dataset_for_nu用于Nu-resnet和NUMO-resnet两个模型的数据进行leave-one-family-out实验
    """
    # dataset = RNA_family_Dataset(family_dir = [
    #           "data/RE-datasetC/16S_rRNA_database",
    #           # "data/RE-datasetC/group_I_intron_database",
    #           "data/RE-datasetC/RNaseP_database",
    #           "data/RE-datasetC/SRP_database",
    #           "data/RE-datasetC/telomerase",
    #           "data/RE-datasetC/tRNA"
    #           ],
    #         rna_fm_dir=r"data/fm_representations/tRNA_rRNA_other_family",
    #         tmRNA_dir=r"data/RE-datasetC/tmRNA",
    #         tmRNA_fm_dir="data/fm_representations/tmRNA",
    #         # tmRNA_dir="",
    #         # tmRNA_fm_dir=""
    #           )

    # dataset = RNA_family_Dataset_for_nu(family_dir=[
    #               "data/RE-datasetC/16S_rRNA_database",
    #               "data/RE-datasetC/group_I_intron_database",
    #               "data/RE-datasetC/RNaseP_database",
    #               "data/RE-datasetC/SRP_database",
    #               "data/RE-datasetC/telomerase",
    #               "data/RE-datasetC/tRNA"
    #               ],
    #                              tmRNA_dir=r"data/RE-datasetC/tmRNA",
    #                              # tmRNA_dir=""
    #                              )

    """
    用于序列判别任务（非跨家族）
    其中RNA_ent_Dataset用于RNAeval模型
    RNA_ent_Dataset_for_nu用于nu-resnet模型和numo-resnet
    """
    # dataset = RNA_ent_Dataset(seq_pos_dir="data/ENT-Dataset/ent_train_real",
    #                           seq_neg_dir="data/ENT-Dataset/ent_train_worst",
    #                           fm_pos_dir="data/fm_representations/pseu_free_real",
    #                           fm_neg_dir="data/fm_representations/pseu_free_worst",
    #                           rna_str_dir="data/ENT-Dataset/ent_train_struc")     # 用于序列判别任务和RNAeval模型

    # dataset = RNA_ent_Dataset_for_nu(seq_pos_dir="data/ENT-Dataset/ent_train_real",
    #                           seq_neg_dir="data/ENT-Dataset/ent_train_worst",
    #                           rna_str_dir="data/ENT-Dataset/ent_train_struc")  # 用于序列判别任务和nu/numo




    main_eval_rna(get_training_args(), dataset_train=dataset)
