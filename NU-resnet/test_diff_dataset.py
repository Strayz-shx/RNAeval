import pandas as pd
from data_preparation_RNA_mat_3d_nt_localized_info_mat_function_PDB_data import data_prep_RNA_mat_3d_nt_localized_info_mat_PDB_data
import torch
import torch.nn as nn
from train_val_test_plot_functions_color_mat_ResNet_18 import create_weighted_sampler, dataloader_prep_with_sampler, dataloader_prep, train_val_ResNet_expert_color_mat, test_ResNet_expert, train_val_figure_plot_function
from ResNet_architecture_grayscale_mat import ResNet_18_grayscale_mat, RNAInception
import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, roc_auc_score, matthews_corrcoef, precision_score, recall_score


# Testing data
Original_RNA_Data_test = pd.read_csv('PDB_data_reliable_negative_samples_testing_data.csv')
#Original_RNA_Data_test = Original_RNA_Data_test.head(10)

Original_RNA_Data_test['RNA_seq_upper']=np.nan
Original_RNA_Data_test['RNA_seq_upper'] = Original_RNA_Data_test['RNA_seq_upper'].astype("string")

num_row_Original_RNA_Data_test = Original_RNA_Data_test.shape[0]

for i in range(num_row_Original_RNA_Data_test):
    Original_RNA_Data_test.at[i, "RNA_seq_upper"] = Original_RNA_Data_test.at[i, "RNA_seq"].upper()

Original_RNA_Data_test = Original_RNA_Data_test[~Original_RNA_Data_test.RNA_seq_upper.str.contains('|'.join(["P"]))].reset_index(drop=True)



# Testing data
test_x_color_mat_np, test_x_nt_localized_info_mat_np, test_y_np, test_x_embedding, test_x_ss = data_prep_RNA_mat_3d_nt_localized_info_mat_PDB_data(Original_RNA_Data = Original_RNA_Data_test, padding_length=410.0)

test_x_color_mat_4D_torch = torch.from_numpy(test_x_color_mat_np)
test_x_color_mat_4D_torch = test_x_color_mat_4D_torch.type(torch.float)

test_x_nt_localized_info_mat_4D_torch = torch.from_numpy(test_x_nt_localized_info_mat_np)
test_x_nt_localized_info_mat_4D_torch = test_x_nt_localized_info_mat_4D_torch.type(torch.float)

test_y_torch = torch.from_numpy(test_y_np)
test_y_torch = test_y_torch.type(torch.long)

test_x_embedding = torch.from_numpy(test_x_embedding)
test_x_embedding = test_x_embedding.type(torch.float)

test_x_ss = torch.from_numpy(test_x_ss)
test_x_ss = test_x_ss.type(torch.float)

#%#
# 参数设置

device_utilized = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NN_architecture_utilized = RNAInception(out_channels=2)

NN_model_utilized = NN_architecture_utilized.to(device_utilized)

# Training Part 2.3: Set hyperparameters, loss function, optimizer, and scheduler.
learning_rate_value_utilized = 0.0001
weight_decay_value_utilized = 0.15
num_epochs_utilized = 60
batch_size_value_utilized = 10
gamma_value_utilized = 0.95

optimizer_utilized = torch.optim.Adam(NN_model_utilized.parameters(), lr=learning_rate_value_utilized, weight_decay=weight_decay_value_utilized)
scheduler_for_optimizer_utilized = torch.optim.lr_scheduler.ExponentialLR(optimizer_utilized, gamma=gamma_value_utilized, last_epoch=-1)

loss_function_utilized = nn.CrossEntropyLoss()

model_name_utilized = 'NU_ResNet_batch_'+str(batch_size_value_utilized)+'_lr_'+str(learning_rate_value_utilized)+\
    '_weightd_'+str(weight_decay_value_utilized)+'_epochs_'+str(num_epochs_utilized)+'_scheduler_gamma_'+\
        str(gamma_value_utilized)+'_PDB_data_tRNA_rRNA.pth'

# test dataloader
testing_data_utilized = dataloader_prep(x_color_mat_stack_4D_tensor = test_x_color_mat_4D_torch,
                                        x_nt_localized_info_mat_stack_4D_tensor = test_x_nt_localized_info_mat_4D_torch,
                                        y_tensor = test_y_torch,
                                        batch_size_value = batch_size_value_utilized,
                                        x_embedding = test_x_embedding,
                                        x_ss=test_x_ss)


#%%
# Load the trained models
# Best val accuracy model

best_val_acc_model_name_utilized = "/home/Shihx/rnadataloder/NU-resnet/output_inception/Best_validation_accuracy_model_NU_ResNet_batch_10_lr_0.0001_weightd_0.15_epochs_60_scheduler_gamma_0.95_PDB_data.pth"

NN_archit_test_best_acc = RNAInception(out_channels=2)

NN_test_model_best_acc = NN_archit_test_best_acc.to(device_utilized)

NN_test_model_best_acc.load_state_dict(torch.load(best_val_acc_model_name_utilized, map_location=device_utilized)['saved_model'])
print('The best validation accuracy model is saved from epoch ', torch.load(best_val_acc_model_name_utilized, map_location=device_utilized)['epoch_num'])

# Best val loss model

best_val_loss_model_name_utilized = "/home/Shihx/rnadataloder/NU-resnet/output_inception/Best_validation_loss_model_NU_ResNet_batch_10_lr_0.0001_weightd_0.15_epochs_60_scheduler_gamma_0.95_PDB_data.pth"

NN_archit_test_best_loss = RNAInception(out_channels=2)

NN_test_model_best_loss = NN_archit_test_best_loss.to(device_utilized)

NN_test_model_best_loss.load_state_dict(torch.load(best_val_loss_model_name_utilized, map_location=device_utilized)['saved_model'])
print('The best validation loss model is saved from epoch ', torch.load(best_val_loss_model_name_utilized, map_location=device_utilized)['epoch_num'])

# Model from last epoch

model_from_last_epoch_name_utilized = "/home/Shihx/rnadataloder/NU-resnet/output_inception/Model_from_last_epoch_NU_ResNet_batch_10_lr_0.0001_weightd_0.15_epochs_60_scheduler_gamma_0.95_PDB_data.pth"

NN_archit_test_model_last_epoch = RNAInception(out_channels=2)

NN_test_model_last_epoch = NN_archit_test_model_last_epoch.to(device_utilized)

NN_test_model_last_epoch.load_state_dict(torch.load(model_from_last_epoch_name_utilized, map_location=device_utilized)['saved_model'])
print('The model from last epoch is saved from epoch ', torch.load(model_from_last_epoch_name_utilized, map_location=device_utilized)['epoch_num'])


loss_mathews_data_best_val_acc, metric_mathews_data_best_val_acc = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_best_acc,
                                                                                           loss_function = loss_function_utilized, model_name = best_val_acc_model_name_utilized,
                                                                                           testinging_data_used = testing_data_utilized, testing_data_name = 'Mathews_testing_data')

print('The metrics of best validation accuracy model on Mathews testing data is as follows.')
print(metric_mathews_data_best_val_acc)

'''
loss_rfam_data_best_val_acc, metric_rfam_data_best_val_acc = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_best_acc, 
                                                                                           loss_function = loss_function_utilized, model_name = best_val_acc_model_name_utilized, 
                                                                                           testinging_data_used = rfam_testing_data_utilized, testing_data_name = 'Rfam_testing_data')

print('The metrics of best validation accuracy model on Rfam testing data is as follows.')
print(metric_rfam_data_best_val_acc)
'''
loss_mathews_data_best_val_loss, metric_mathews_data_best_val_loss = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_best_loss,
                                                                                           loss_function = loss_function_utilized, model_name = best_val_loss_model_name_utilized,
                                                                                           testinging_data_used = testing_data_utilized, testing_data_name = 'Mathews_testing_data')

print('The metrics of best validation loss model on Mathews testing data is as follows.')
print(metric_mathews_data_best_val_loss)
'''
loss_rfam_data_best_val_loss, metric_rfam_data_best_val_loss = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_best_loss, 
                                                                                           loss_function = loss_function_utilized, model_name = best_val_loss_model_name_utilized, 
                                                                                           testinging_data_used = rfam_testing_data_utilized, testing_data_name = 'Rfam_testing_data')

print('The metrics of best validation loss model on Rfam testing data is as follows.')
print(metric_rfam_data_best_val_loss)
'''
loss_mathews_data_model_last_epoch, metric_mathews_data_model_last_epoch = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_last_epoch,
                                                                                           loss_function = loss_function_utilized, model_name = model_from_last_epoch_name_utilized,
                                                                                           testinging_data_used = testing_data_utilized, testing_data_name = 'Mathews_testing_data')

print('The metrics of the model from last epoch on Mathews testing data is as follows.')
print(metric_mathews_data_model_last_epoch)
'''
loss_rfam_data_model_last_epoch, metric_rfam_data_model_last_epoch = test_ResNet_expert(device = device_utilized, NN_model = NN_test_model_last_epoch, 
                                                                                           loss_function = loss_function_utilized, model_name = model_from_last_epoch_name_utilized, 
                                                                                           testinging_data_used = rfam_testing_data_utilized, testing_data_name = 'Rfam_testing_data')

print('The metrics of the model from last epoch on Rfam testing data is as follows.')
print(metric_rfam_data_model_last_epoch)
'''


def classification_model_performance_metrics(all_preds_score_positive_sample, all_preds_label, all_true_label):
    classification_metric = {'model_accuracy': float("-inf"),
                             'model_auc_roc': float("-inf"),
                             'model_auc_roc_check': float("-inf"),
                             'model_mcc': float("-inf"),
                             'model_precision': float("-inf"),
                             'model_recall': float("-inf"),
                             'model_specificity': float("-inf")}

    print(classification_report(y_true=all_true_label, y_pred=all_preds_label))

    print('The confusion matrix is as follows.')
    confusion_mat_obtained = confusion_matrix(y_true=all_true_label, y_pred=all_preds_label)
    print(confusion_mat_obtained)

    accuracy_obtained = accuracy_score(y_true=all_true_label, y_pred=all_preds_label, normalize=True)
    print("The classification accuracy is {}.".format(accuracy_obtained))
    classification_metric['model_accuracy'] = accuracy_obtained

    false_positive_r, true_positive_r, threshold_obtained = roc_curve(y_true=all_true_label,
                                                                      y_score=all_preds_score_positive_sample,
                                                                      pos_label=1)
    auc_roc_obtained_check = auc(false_positive_r, true_positive_r)
    auc_roc_obtained = roc_auc_score(y_true=all_true_label, y_score=all_preds_score_positive_sample, sample_weight=None)
    print("The auc roc is {}.".format(auc_roc_obtained))
    print("The auc roc check is {}.".format(auc_roc_obtained_check))
    classification_metric['model_auc_roc'] = auc_roc_obtained
    classification_metric['model_auc_roc_check'] = auc_roc_obtained_check

    mcc_obtained = matthews_corrcoef(y_true=all_true_label, y_pred=all_preds_label, sample_weight=None)
    print("The mcc is {}.".format(mcc_obtained))
    classification_metric['model_mcc'] = mcc_obtained

    precision_obtained = precision_score(y_true=all_true_label, y_pred=all_preds_label, pos_label=1, average='binary',
                                         sample_weight=None)
    print("The precision is {}.".format(precision_obtained))
    classification_metric['model_precision'] = precision_obtained

    recall_obtained = recall_score(y_true=all_true_label, y_pred=all_preds_label, pos_label=1, average='binary',
                                   sample_weight=None)
    print("The recall is {}.".format(recall_obtained))
    classification_metric['model_recall'] = recall_obtained

    specificity_obtained = confusion_mat_obtained[0, 0] / (confusion_mat_obtained[0, 1] + confusion_mat_obtained[0, 0])
    print("The specificity is {}.".format(specificity_obtained))
    classification_metric['model_specificity'] = specificity_obtained

    return classification_metric