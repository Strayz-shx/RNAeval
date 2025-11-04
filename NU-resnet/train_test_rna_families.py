import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, roc_auc_score, matthews_corrcoef, precision_score, recall_score
import matplotlib.pyplot as plt
from train_val_test_plot_functions_color_mat_ResNet_18 import create_weighted_sampler, dataloader_prep_with_sampler, dataloader_prep, train_val_ResNet_expert_color_mat, test_ResNet_expert, train_val_figure_plot_function
import torch.nn as nn
import pandas as pd
from data_preparation_RNA_mat_3d_nt_localized_info_mat_function_PDB_data import data_prep_RNA_mat_3d_nt_localized_info_mat_PDB_data, data_prep_RNA_for_tRNA_rRNA
from ResNet_architecture_grayscale_mat import RNAInception# , ResNet_18_grayscale_mat

def RNA_family_testing_function(test_data_file, device_utilized_para, loss_function_utilized_para, testing_data_name_para, NN_test_model_best_acc_para, best_val_acc_model_name_para, NN_test_model_best_loss_para, best_val_loss_model_name_para, NN_test_model_last_epoch_para, model_from_last_epoch_name_para):

    # 1.1 Read the data, select the utilized columns, and rename the column name.
    # Testing data
    Original_RNA_Data_test = pd.read_csv(test_data_file)
    #Original_RNA_Data_test = Original_RNA_Data_test.head(10)
    '''
    Original_RNA_Data_test['RNA_seq_upper']=np.nan
    Original_RNA_Data_test['RNA_seq_upper'] = Original_RNA_Data_test['RNA_seq_upper'].astype("string")

    num_row_Original_RNA_Data_test = Original_RNA_Data_test.shape[0]

    for i in range(num_row_Original_RNA_Data_test):
        Original_RNA_Data_test.at[i, "RNA_seq_upper"] = Original_RNA_Data_test.at[i, "RNA_seq"].upper()
    '''
    Original_RNA_Data_test = Original_RNA_Data_test[~Original_RNA_Data_test.RNA_seq_upper.str.contains('|'.join(["P"]))].reset_index(drop=True)

    # Testing data
    test_x_color_mat_np, test_x_nt_localized_info_mat_np, test_y_np, test_x_embedding, test_x_ss  = data_prep_RNA_for_tRNA_rRNA(Original_RNA_Data = Original_RNA_Data_test, padding_length=410.0)

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

    # Training Part 2.4: Prepare the Dataloader

    testing_data_utilized = dataloader_prep(x_color_mat_stack_4D_tensor = test_x_color_mat_4D_torch,
                                            x_nt_localized_info_mat_stack_4D_tensor = test_x_nt_localized_info_mat_4D_torch,
                                            y_tensor = test_y_torch,
                                            batch_size_value = 1,
                                            x_embedding=test_x_embedding,
                                            x_ss=test_x_ss)

    # Test trained models on testing data

    loss_best_val_acc, metric_best_val_acc = test_ResNet_expert(device = device_utilized_para, NN_model = NN_test_model_best_acc_para,
                                                                                            loss_function = loss_function_utilized_para, model_name = best_val_acc_model_name_para,
                                                                                            testinging_data_used = testing_data_utilized, testing_data_name = testing_data_name_para)

    print('The metrics of best validation accuracy model is as follows.')
    print(metric_best_val_acc)

    loss_best_val_loss, metric_best_val_loss = test_ResNet_expert(device = device_utilized_para, NN_model = NN_test_model_best_loss_para,
                                                                                            loss_function = loss_function_utilized_para, model_name = best_val_loss_model_name_para,
                                                                                            testinging_data_used = testing_data_utilized, testing_data_name = testing_data_name_para)

    print('The metrics of best validation loss model is as follows.')
    print(metric_best_val_loss)

    loss_model_last_epoch, metric_model_last_epoch = test_ResNet_expert(device = device_utilized_para, NN_model = NN_test_model_last_epoch_para,
                                                                                            loss_function = loss_function_utilized_para, model_name = model_from_last_epoch_name_para,
                                                                                            testinging_data_used = testing_data_utilized, testing_data_name = testing_data_name_para)

    print('The metrics of the model from last epoch is as follows.')
    print(metric_model_last_epoch)

    return loss_best_val_acc, metric_best_val_acc, loss_best_val_loss, metric_best_val_loss, loss_model_last_epoch, metric_model_last_epoch


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


def test_ResNet_expert(device, NN_model, loss_function, model_name, testinging_data_used, testing_data_name):
    all_preds_test = []
    all_preds_score_test = []
    all_labels_test = []
    NN_model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, mat_label in enumerate(testinging_data_used):
            color_mat_test, nt_localized_info_mat_test, label_test, embedding, ss = mat_label[0].to(device), mat_label[1].to(device), \
            mat_label[2].to(device), mat_label[3].to(device), mat_label[4].to(device)
            pred_test = NN_model(embedding, ss)
            loss = loss_function(pred_test, label_test)
            pred_test_score = torch.nn.functional.softmax(pred_test.cpu().detach(), dim=1)

            all_preds_score_test.append(pred_test_score.numpy()[:, 1])

            all_preds_test.append(np.argmax(pred_test.cpu().detach().numpy(), axis=1))
            all_labels_test.append(label_test.cpu().detach().numpy())
            test_loss += loss.item()

    print("The performance of the trained model ", model_name, " based on the ", testing_data_name, " is as follows.")

    all_preds_score_test = np.concatenate(all_preds_score_test).ravel()
    all_preds_test = np.concatenate(all_preds_test).ravel()
    all_labels_test = np.concatenate(all_labels_test).ravel()

    classification_metric_test = classification_model_performance_metrics(
        all_preds_score_positive_sample=all_preds_score_test,
        all_preds_label=all_preds_test,
        all_true_label=all_labels_test)

    test_loss_this_data_set = test_loss / len(testinging_data_used)
    print('The testing loss value is {}.'.format(test_loss_this_data_set))

    return test_loss_this_data_set, classification_metric_test


# Training data
Original_RNA_Data_train = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_tRNA_rRNA_training_data.csv')


# Validation data
Original_RNA_Data_val = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_tRNA_rRNA_validation_data.csv')

#%%
# Training data prep
train_x_color_mat_np, train_x_nt_localized_info_mat_np, train_y_np, train_x_embedding, train_x_ss = data_prep_RNA_for_tRNA_rRNA(Original_RNA_Data = Original_RNA_Data_train, padding_length=410.0)

train_x_color_mat_4D_torch = torch.from_numpy(train_x_color_mat_np)
train_x_color_mat_4D_torch = train_x_color_mat_4D_torch.type(torch.float)

train_x_nt_localized_info_mat_4D_torch = torch.from_numpy(train_x_nt_localized_info_mat_np)
train_x_nt_localized_info_mat_4D_torch = train_x_nt_localized_info_mat_4D_torch.type(torch.float)

train_y_torch = torch.from_numpy(train_y_np)
train_y_torch = train_y_torch.type(torch.long)

train_x_embedding = torch.from_numpy(train_x_embedding)
train_x_embedding = train_x_embedding.type(torch.float)

train_x_ss = torch.from_numpy(train_x_ss)
train_x_ss = train_x_ss.type(torch.float)

# Validation data prep
val_x_color_mat_np, val_x_nt_localized_info_mat_np, val_y_np, val_x_embedding, val_x_ss = data_prep_RNA_for_tRNA_rRNA(Original_RNA_Data = Original_RNA_Data_val, padding_length=410.0)

val_x_color_mat_4D_torch = torch.from_numpy(val_x_color_mat_np)
val_x_color_mat_4D_torch = val_x_color_mat_4D_torch.type(torch.float)

val_x_nt_localized_info_mat_4D_torch = torch.from_numpy(val_x_nt_localized_info_mat_np)
val_x_nt_localized_info_mat_4D_torch = val_x_nt_localized_info_mat_4D_torch.type(torch.float)

val_y_torch = torch.from_numpy(val_y_np)
val_y_torch = val_y_torch.type(torch.long)

val_x_embedding = torch.from_numpy(val_x_embedding)
val_x_embedding = val_x_embedding.type(torch.float)

val_x_ss = torch.from_numpy(val_x_ss)
val_x_ss = val_x_ss.type(torch.float)
'''
'''

#%%

sampler_weight_data_loader_utilized = create_weighted_sampler(y_numpy = train_y_np)

device_utilized = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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



# Training Part 2.4: Prepare the Dataloader

training_data_utilized = dataloader_prep_with_sampler(x_color_mat_stack_4D_tensor = train_x_color_mat_4D_torch,
                                                      x_nt_localized_info_mat_stack_4D_tensor = train_x_nt_localized_info_mat_4D_torch,
                                                      y_tensor = train_y_torch,
                                                      batch_size_value = batch_size_value_utilized,
                                                      sampler_applied = sampler_weight_data_loader_utilized,
                                                      x_embedding=train_x_embedding,
                                                      x_ss=train_x_ss)


val_data_utilized = dataloader_prep(x_color_mat_stack_4D_tensor = val_x_color_mat_4D_torch,
                                    x_nt_localized_info_mat_stack_4D_tensor = val_x_nt_localized_info_mat_4D_torch,
                                    y_tensor = val_y_torch,
                                    batch_size_value = batch_size_value_utilized,
                                    x_embedding=val_x_embedding,
                                    x_ss=val_x_ss)


#%%
train_val_record_obtained, saved_models_metrics_obtained = train_val_ResNet_expert_color_mat(device = device_utilized,
                                                                                             NN_model = NN_model_utilized,
                                                                                             num_epochs = num_epochs_utilized,
                                                                                             optimizer = optimizer_utilized,
                                                                                             scheduler_for_optimizer = scheduler_for_optimizer_utilized,
                                                                                             loss_function = loss_function_utilized,
                                                                                             model_name = model_name_utilized,
                                                                                             training_data_used = training_data_utilized,
                                                                                             val_data_used = val_data_utilized)

print('The metrics of best validation accuracy model is as follows.')
print(saved_models_metrics_obtained['best_val_acc_model'])

print('The metrics of best validation loss model is as follows.')
print(saved_models_metrics_obtained['best_val_loss_model'])

print('The metrics of the model from last epoch is as follows.')
print(saved_models_metrics_obtained['last_epoch_model'])


# Save the results from training into csv files.

train_val_record_file_name = 'NUResNet_train_val_record_batch_'+str(batch_size_value_utilized)+'_lr_'+str(learning_rate_value_utilized)+\
    '_weightd_'+str(weight_decay_value_utilized)+'_epochs_'+str(num_epochs_utilized)+'_scheduler_gamma_'+\
        str(gamma_value_utilized)+'PDB_data_tRNA_rRNA.csv'

df_train_val_record = pd.DataFrame.from_dict(train_val_record_obtained)

df_train_val_record.to_csv(train_val_record_file_name, index=False)

saved_models_metrics_file_name = 'NUResNet_saved_models_metrics_batch_'+str(batch_size_value_utilized)+'_lr_'+str(learning_rate_value_utilized)+\
    '_weightd_'+str(weight_decay_value_utilized)+'_epochs_'+str(num_epochs_utilized)+'_scheduler_gamma_'+\
        str(gamma_value_utilized)+'PDB_data_tRNA_rRNA.csv'

df_saved_models_metrics = pd.DataFrame.from_dict(saved_models_metrics_obtained)

df_saved_models_metrics.to_csv(saved_models_metrics_file_name, index=False)


#%%
# Load the trained models
# Best val accuracy model

best_val_acc_model_name_utilized = 'Best_validation_accuracy_model_'+model_name_utilized

NN_archit_test_best_acc = RNAInception(out_channels=2)

NN_test_model_best_acc = NN_archit_test_best_acc.to(device_utilized)

NN_test_model_best_acc.load_state_dict(torch.load(best_val_acc_model_name_utilized, map_location=device_utilized)['saved_model'])
print('The best validation accuracy model is saved from epoch ', torch.load(best_val_acc_model_name_utilized, map_location=device_utilized)['epoch_num'])

# Best val loss model

best_val_loss_model_name_utilized = 'Best_validation_loss_model_'+model_name_utilized

NN_archit_test_best_loss = RNAInception(out_channels=2)

NN_test_model_best_loss = NN_archit_test_best_loss.to(device_utilized)

NN_test_model_best_loss.load_state_dict(torch.load(best_val_loss_model_name_utilized, map_location=device_utilized)['saved_model'])
print('The best validation loss model is saved from epoch ', torch.load(best_val_loss_model_name_utilized, map_location=device_utilized)['epoch_num'])

# Model from last epoch

model_from_last_epoch_name_utilized = 'Model_from_last_epoch_'+model_name_utilized

NN_archit_test_model_last_epoch = RNAInception(out_channels=2)

NN_test_model_last_epoch = NN_archit_test_model_last_epoch.to(device_utilized)

NN_test_model_last_epoch.load_state_dict(torch.load(model_from_last_epoch_name_utilized, map_location=device_utilized)['saved_model'])
print('The model from last epoch is saved from epoch ', torch.load(model_from_last_epoch_name_utilized, map_location=device_utilized)['epoch_num'])


#%%
# 1. GroupIintron
print('GroupIintron: ')

GroupI_loss_b_v_a, GroupI_metric_b_v_a, GroupI_loss_b_v_l, GroupI_metric_b_v_l, GroupI_loss_model_l_e, GroupI_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_GroupIintron_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='GroupIintron',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)

#%%
# 2. GroupIIintron
print('GroupIIintron: ')

GroupII_loss_b_v_a, GroupII_metric_b_v_a, GroupII_loss_b_v_l, GroupII_metric_b_v_l, GroupII_loss_model_l_e, GroupII_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_GroupIIintron_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='GroupIIintron',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)

#%%
# 3. SRPRNA
print('SRPRNA: ')

SRP_loss_b_v_a, SRP_metric_b_v_a, SRP_loss_b_v_l, SRP_metric_b_v_l, SRP_loss_model_l_e, SRP_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_SRPRNA_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='SRPRNA',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)

#%%
# 4. HairpinRibozyme
print('HairpinRibozyme: ')

HR_loss_b_v_a, HR_metric_b_v_a, HR_loss_b_v_l, HR_metric_b_v_l, HR_loss_model_l_e, HR_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_HairpinRibozyme_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='HairpinRibozyme',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)

#%%
# 5. HammerheadRibozyme
print('HammerheadRibozyme: ')

HHR_loss_b_v_a, HHR_metric_b_v_a, HHR_loss_b_v_l, HHR_metric_b_v_l, HHR_loss_model_l_e, HHR_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_HammerheadRibozyme_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='HammerheadRibozyme',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)





#%%
# 6. otherRibozyme
print('otherRibozyme: ')

OR_loss_b_v_a, OR_metric_b_v_a, OR_loss_b_v_l, OR_metric_b_v_l, OR_loss_model_l_e, OR_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_otherRibozyme_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='otherRibozyme',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)

#%%
# 7. ViralandPhage
print('ViralandPhage: ')

VP_loss_b_v_a, VP_metric_b_v_a, VP_loss_b_v_l, VP_metric_b_v_l, VP_loss_model_l_e, VP_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_ViralandPhage_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='ViralandPhage',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)

#%%
# 8. SmallnuclearRNA
print('SmallnuclearRNA: ')

SN_loss_b_v_a, SN_metric_b_v_a, SN_loss_b_v_l, SN_metric_b_v_l, SN_loss_model_l_e, SN_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_SmallnuclearRNA_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='SmallnuclearRNA',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)

#%%
# 9. InternalRibosomeEntrySite
print('InternalRibosomeEntrySite: ')

IRES_loss_b_v_a, IRES_metric_b_v_a, IRES_loss_b_v_l, IRES_metric_b_v_l, IRES_loss_model_l_e, IRES_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_InternalRibosomeEntrySite_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='InternalRibosomeEntrySite',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)


#%%
# 10. RNasePRNA
print('RNasePRNA: ')

RNaseP_loss_b_v_a, RNaseP_metric_b_v_a, RNaseP_loss_b_v_l, RNaseP_metric_b_v_l, RNaseP_loss_model_l_e, RNaseP_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_RNasePRNA_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='RNasePRNA',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)


#%%
# 11. otherRNA
print('otherRNA: ')

O_loss_b_v_a, O_metric_b_v_a, O_loss_b_v_l, O_metric_b_v_l, O_loss_model_l_e, O_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_otherRNA_with_info.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='otherRNA',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)






#%%
# concatenate the data from different RNA families
print('concatenate the data from different RNA families: ')

GroupIintron_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_GroupIintron_with_info.csv')

GroupIIintron_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_GroupIIintron_with_info.csv')

SRP_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_SRPRNA_with_info.csv')

HairpinRibozyme_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_HairpinRibozyme_with_info.csv')

HammerheadRibozyme_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_HammerheadRibozyme_with_info.csv')

otherRibozyme_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_otherRibozyme_with_info.csv')

ViralandPhage_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_ViralandPhage_with_info.csv')

SmallnuclearRNA_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_SmallnuclearRNA_with_info.csv')

InternalRibosomeEntrySite_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_InternalRibosomeEntrySite_with_info.csv')

RNasePRNA_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_RNasePRNA_with_info.csv')

concatenated_RNA_Data = pd.concat([GroupIintron_RNA_Data,
                                   GroupIIintron_RNA_Data,
                                   SRP_RNA_Data,
                                   HairpinRibozyme_RNA_Data,
                                   HammerheadRibozyme_RNA_Data,
                                   otherRibozyme_RNA_Data,
                                   ViralandPhage_RNA_Data,
                                   SmallnuclearRNA_RNA_Data,
                                   InternalRibosomeEntrySite_RNA_Data,
                                   RNasePRNA_RNA_Data], axis=0, ignore_index=True)

concatenated_RNA_Data.to_csv('/home/Shihx/rnadataloder/RNA_familiy_data/concatenated_different_RNA_families_testing_data.csv', index=False)

concat_loss_b_v_a, concat_metric_b_v_a, concat_loss_b_v_l, concat_metric_b_v_l, concat_loss_model_l_e, concat_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/concatenated_different_RNA_families_testing_data.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='/home/Shihx/rnadataloder/RNA_familiy_data/concatenated_RNA_data',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)


#%%
# concatenate the data from different RNA families and other RNA
print('concatenate the data from different RNA families and other RNA: ')

otherRNA_RNA_Data = pd.read_csv('/home/Shihx/rnadataloder/RNA_familiy_data/PDB_data_otherRNA_with_info.csv')

concatenated_different_RNA_families_otherRNA = pd.concat([concatenated_RNA_Data,
                                                          otherRNA_RNA_Data], axis=0, ignore_index=True)

concatenated_different_RNA_families_otherRNA.to_csv('/home/Shihx/rnadataloder/RNA_familiy_data/concatenated_different_RNA_families_otherRNA_testing_data.csv', index=False)

concat_2_loss_b_v_a, concat_2_metric_b_v_a, concat_2_loss_b_v_l, concat_2_metric_b_v_l, concat_2_loss_model_l_e, concat_2_metric_model_l_e = RNA_family_testing_function(test_data_file='/home/Shihx/rnadataloder/RNA_familiy_data/concatenated_different_RNA_families_otherRNA_testing_data.csv',
                                                                                                                                                             device_utilized_para=device_utilized,
                                                                                                                                                             loss_function_utilized_para=loss_function_utilized,
                                                                                                                                                             testing_data_name_para='/home/Shihx/rnadataloder/RNA_familiy_data/concatenated_data_from_different_RNA_families_otherRNA',
                                                                                                                                                             NN_test_model_best_acc_para=NN_test_model_best_acc,
                                                                                                                                                             best_val_acc_model_name_para=best_val_acc_model_name_utilized,
                                                                                                                                                             NN_test_model_best_loss_para=NN_test_model_best_loss,
                                                                                                                                                             best_val_loss_model_name_para=best_val_loss_model_name_utilized,
                                                                                                                                                             NN_test_model_last_epoch_para=NN_test_model_last_epoch,
                                                                                                                                                             model_from_last_epoch_name_para=model_from_last_epoch_name_utilized)


#%%

figure_name_utilized = 'NU_ResNet_batch_'+str(batch_size_value_utilized)+'_lr_'+str(learning_rate_value_utilized)+\
    '_weightd_'+str(weight_decay_value_utilized)+'_epochs_'+str(num_epochs_utilized)+'_scheduler_gamma_'+\
        str(gamma_value_utilized)+'_PDB_data_tRNA_rRNA'


train_val_figure_plot_function(figure_name = figure_name_utilized,
                               loss_value_train = train_val_record_obtained['loss_train'],
                               loss_value_val = train_val_record_obtained['loss_val'],
                               accuracy_train = train_val_record_obtained['acc_train'],
                               accuracy_val = train_val_record_obtained['acc_val'],
                               auc_roc_train = train_val_record_obtained['aucroc_train'],
                               auc_roc_val = train_val_record_obtained['aucroc_val'],
                               auc_roc_train_check = train_val_record_obtained['aucroc_train_check'],
                               auc_roc_val_check = train_val_record_obtained['aucroc_val_check'])


