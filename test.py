import torch

import matplotlib.pyplot as plt
from dataset import RNASeqDataset, Nu_res_Dataset, RNA_family_Dataset, RNA_family_Dataset_for_nu, RNA_ent_Dataset, RNA_ent_Dataset_for_nu
from dataset_cnn import RNA_score_dataset_multi_result
from torch.utils.data import DataLoader
from model import RNAInception, RNAInception_modify, RNACnn_trans, BasicBlock, ResNet_18_grayscale_mat, ResNet_18_pair_grayscale_mat_nt_localized_info_mat
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,matthews_corrcoef, roc_auc_score, confusion_matrix


def test_subseq(dataset, device, model_path):

    dataloader_test = DataLoader(dataset, batch_size=4, num_workers=4)

    model = RNAInception(out_channels=512)
    # model = RNAInception_modify(out_channels=2)
    # model = RNACnn_trans(BasicBlock, layers=[2, 2, 2, 2], layers_struc=[2, 2, 2, 2],
    #                      out_channels=512).to(device)
    # model = torch.nn.DataParallel(model, device_ids=[2, 3])
    # model = ResNet_18_grayscale_mat()
    # model = ResNet_18_pair_grayscale_mat_nt_localized_info_mat()


    model = torch.nn.DataParallel(model, device_ids=[1, 0])
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    # test_mae, test_mse, teat_rmse, pearsonr, spearmanr = test_with_correlation(model, dataloader_test)
    # print("test_mae{:.4f}\n"
    #       "test_mse{:.4f}\n"
    #       "test_rmse{:.4f}\n"
    #       .format(test_mae, test_mse, teat_rmse))

    # test_nuresnet(model, dataloader_test)
    # test_numoresnet(model, dataloader_test)
    test(model, dataloader_test)


def test_numoresnet(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            color_mat, nt_localized_mat, label = batch
            color_mat, nt_localized_mat, label = color_mat.to(device), nt_localized_mat.to(device), label.to(device)

            label = label.long()

            outputs = model(color_mat, nt_localized_mat)
            loss = F.cross_entropy(outputs, label)

            predicted = torch.argmax(outputs, 1)

            all_predictions.append(predicted.cpu())
            all_labels.append(label.cpu())

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # 拼接所有批次的预测和真实标签
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_predictions)
    auroc = roc_auc_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mcc:{mcc:.4f}")
    print(f"AUROC: {auroc:.4f}" if auroc is not None else "AUROC: N/A")
    print(f"Specificity: {specificity:.4f}" if specificity is not None else "Specificity: N/A")


    print(avg_loss)


def test_nuresnet(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            color_mat, nt_localized_mat, label = batch
            color_mat, nt_localized_mat, label = color_mat.to(device), nt_localized_mat.to(device), label.to(device)

            label = label.long()

            outputs = model(color_mat)
            loss = F.cross_entropy(outputs, label)

            predicted = torch.argmax(outputs, 1)

            all_predictions.append(predicted.cpu())
            all_labels.append(label.cpu())

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # 拼接所有批次的预测和真实标签
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_predictions)
    auroc = roc_auc_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mcc:{mcc:.4f}")
    print(f"AUROC: {auroc:.4f}" if auroc is not None else "AUROC: N/A")
    print(f"Specificity: {specificity:.4f}" if specificity is not None else "Specificity: N/A")

    print(avg_loss)


def test(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            struc, x, labels = batch
            # Move data to device (GPU/CPU)
            x, struc, labels = x.float().to(device), struc.float().to(device), labels.float().to(
                device)

            labels = labels.long()

            outputs = model(struc, x)
            loss = F.cross_entropy(outputs, labels)

            predicted = torch.argmax(outputs, 1)

            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # 拼接所有批次的预测和真实标签
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_predictions)
    auroc = roc_auc_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mcc:{mcc:.4f}")
    print(f"AUROC: {auroc:.4f}" if auroc is not None else "AUROC: N/A")
    print(f"Specificity: {specificity:.4f}" if specificity is not None else "Specificity: N/A")

    print(avg_loss)



if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:1")

    dataset = RNASeqDataset(bpseq_pos_dir=r"data/RE-datasetB/TestSetB",
                            bpseq_neg_dir=r"data/RE-datasetB/pre_Btestset",  # data/RNA_diff/pretestB_middle data/RNA_diff/pretestB_high
                            rna_fm_dir="data/fm_representations/TestB")

    # dataset = Nu_res_Dataset(bpseq_pos_dir=r"data/RE-datasetA/TestSetA",
    #                          bpseq_neg_dir=r"data/RE-datasetA/pre_Atestset")

    # dataset = RNA_family_Dataset(family_dir=[#"data/RE-datasetC/telomerase",
    #                                          # "data/RE-datasetC/16S_rRNA_database",
    #                                          # "data/RE-datasetC/tRNA",
    #                                          "data/RE-datasetC/group_I_intron_database",
    #                                          # "data/RE-datasetC/RNaseP_database",
    #                                          # "data/RE-datasetC/SRP_database"
    #                               ],
    #                              # tmRNA_fm_dir='data/fm_representations/tmRNA',
    #                              # tmRNA_dir='data/RE-datasetC/tmRNA',
    #                              rna_fm_dir=r"data/fm_representations/tRNA_rRNA_other_family",
    #                              # rna_fm_dir=r""
    #                              tmRNA_dir="",
    #                              tmRNA_fm_dir=""
    #                               )

    # dataset = RNA_family_Dataset_for_nu(family_dir=[# "data/RE-datasetC/16S_rRNA_database",
    #                                          # "data/RE-datasetC/group_I_intron_database",
    #                                          # "data/RE-datasetC/RNaseP_database",
    #                                          # "data/RE-datasetC/SRP_database",
    #                                          # "data/RE-datasetC/telomerase",
    #                                          # "data/RE-datasetC/tRNA",
    #                                          ],
    #                              # tmRNA_dir = "",
    #                              tmRNA_dir=r"data/RE-datasetC/tmRNA",
    #                              )
    # dataset = RNA_ent_Dataset(seq_pos_dir="data/NU-Dataset/free_test_real",
    #                           seq_neg_dir="data/NU-Dataset/free_test_worst",
    #                           fm_pos_dir="data/fm_representations/NU_resnet",
    #                           fm_neg_dir="data/fm_representations/NU_resnet_generated",
    #                           rna_str_dir="data/NU-Dataset/free_test_struc")

    # dataset = RNA_ent_Dataset(seq_pos_dir="/home/Shihx/ENTdataset/ent_test_real",
    #                           seq_neg_dir="/home/Shihx/ENTdataset/ent_test_worst",
    #                           fm_pos_dir="data/fm_representations/pseu_free_real",
    #                           fm_neg_dir="data/fm_representations/pseu_free_worst",
    #                           rna_str_dir="/home/Shihx/ENTdataset/ent_test_struc")

    # dataset = RNA_ent_Dataset_for_nu(seq_pos_dir="data/NU-Dataset/free_test_real",
    #                                  seq_neg_dir="data/NU-Dataset/free_test_worst",
    #                                  rna_str_dir="data/NU-Dataset/free_test_struc")

    # dataset = RNA_ent_Dataset_for_nu(seq_pos_dir="data/ENT-Dataset/ent_test_real",
    #                                  seq_neg_dir="data/ENT-Dataset/ent_test_worst",
    #                                  rna_str_dir="data/ENT-Dataset/ent_test_struc")

    # dataset = RNA_family_Dataset_for_nu(family_dir=["data/RE-datasetC/16S_rRNA_database"], tmRNA_dir="")

    model_path = r"checkpoint/4_layer.pth"

    test_subseq(dataset, device, model_path)



