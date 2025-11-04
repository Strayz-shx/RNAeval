
import numpy as np
import pandas as pd
from pseudoknot_free_PDB_train_data_V2 import entrna_main_PDB_train_data
from train_val_test_plot_functions_color_mat_ResNet_18 import classification_model_performance_metrics

# 1. Load the data
Original_RNA_Data_train = pd.read_csv('pseu_free_test.csv')  # ent_test.csv

# 2. Select columns
Original_RNA_Data_train = Original_RNA_Data_train[['RNA_name', 'RNA_seq', 'RNA_secondary_structure', 'RNA_generated_seq']]

#Original_RNA_Data_train = Original_RNA_Data_train.rename(columns={'MCC_RNAfold' : 'MCC'})

# 3. Normalize the MCC
#Original_RNA_Data_train['Normalized_MCC']=np.nan

#Original_RNA_Data_train['Normalized_MCC'] = (Original_RNA_Data_train['MCC']+1)/2

# 4. Obtain the pair of RNA sequence and correct RNA secondary structure  
Original_RNA_Data_actual_RNA_structure = Original_RNA_Data_train[['RNA_name', 'RNA_seq', 'RNA_secondary_structure']].copy(deep=True)

#Original_RNA_Data_actual_RNA_structure['Normalized_MCC']=np.nan

#Original_RNA_Data_actual_RNA_structure['Normalized_MCC']=1

# 5. Obtain the pair of RNA sequence and predicted RNA secondary structure whose normalized MCC less than 1.
Original_RNA_Data_predicted_RNA_structure = Original_RNA_Data_train[['RNA_name', 'RNA_generated_seq', 'RNA_secondary_structure']].copy(deep=True)

#Original_RNA_Data_predicted_RNA_structure = Original_RNA_Data_predicted_RNA_structure[Original_RNA_Data_predicted_RNA_structure['Normalized_MCC'] < 1]

# 6. Obtain the label and rename the corresponding column name
Original_RNA_Data_actual_RNA_structure['RNA_Label'] = 1

Original_RNA_Data_predicted_RNA_structure['RNA_Label'] = 0

Original_RNA_Data_actual_RNA_structure = Original_RNA_Data_actual_RNA_structure.rename(columns={"RNA_seq":"RNA_sequence", "RNA_secondary_structure":"RNA_sec_structure"})

Original_RNA_Data_predicted_RNA_structure = Original_RNA_Data_predicted_RNA_structure.rename(columns={"RNA_generated_seq":"RNA_sequence", "RNA_secondary_structure":"RNA_sec_structure"})

# 7. Concatenate two data sets
Original_RNA_Data_combined = pd.concat([Original_RNA_Data_actual_RNA_structure, Original_RNA_Data_predicted_RNA_structure], axis=0, ignore_index=True) 

# 8. Obtain the label for each sample  
#Original_RNA_Data_combined['RNA_Label'] = np.where(Original_RNA_Data_combined['Normalized_MCC'] < 1, 0, 1)

# 9. Obtain the features for ENTRNA
Original_RNA_Data_combined['ENTRNA_foldability']=np.nan
Original_RNA_Data_combined['train_accuracy']=np.nan
Original_RNA_Data_combined['pred_Label']=np.nan
#Original_RNA_Data_combined['expected_accuracy']=np.nan
#Original_RNA_Data_combined['fe_per']=np.nan

num_row_Original_RNA_Data_combined = Original_RNA_Data_combined.shape[0]

for i in range(num_row_Original_RNA_Data_combined):
    #ENTRNA_features = extract_features_pseudoknot_free(seq=Original_RNA_Data_combined.at[i, 'RNA_sequence'], sec_str=Original_RNA_Data_combined.at[i, 'RNA_sec_structure'])
    foldability_ENTRNA, train_acc_ENTRNA = entrna_main_PDB_train_data(seq = Original_RNA_Data_combined.at[i, 'RNA_sequence'].upper(), sec_str = Original_RNA_Data_combined.at[i, 'RNA_sec_structure'])
    Original_RNA_Data_combined.at[i, 'ENTRNA_foldability'] = foldability_ENTRNA
    Original_RNA_Data_combined.at[i, 'train_accuracy'] = train_acc_ENTRNA
    #Original_RNA_Data_combined.at[i, 'ensemble_diversity'] = ENTRNA_features.at[0, 'ensemble_diversity']
    #Original_RNA_Data_combined.at[i, 'expected_accuracy'] = ENTRNA_features.at[0, 'expected_accuracy']
    #Original_RNA_Data_combined.at[i, 'fe_per'] = ENTRNA_features.at[0, 'fe_per']

# 10. Obtain the predicted label from ENTRNA    
Original_RNA_Data_combined['pred_Label'] = np.where(Original_RNA_Data_combined['ENTRNA_foldability'] <= 0.5, 0, 1)

# 11. Obtain the metrics of classification performance

np_True_label = Original_RNA_Data_combined['RNA_Label'].to_numpy()
np_Pred_label = Original_RNA_Data_combined['pred_Label'].to_numpy()
np_score_positive_sample = Original_RNA_Data_combined['ENTRNA_foldability'].to_numpy()


ENTRNA_classification_metric = classification_model_performance_metrics(all_preds_score_positive_sample=np_score_positive_sample, 
                                                                        all_preds_label=np_Pred_label, 
                                                                        all_true_label=np_True_label)

print(ENTRNA_classification_metric)
# 10. Shuffle the data
#Original_RNA_Data_combined = Original_RNA_Data_combined.sample(frac=1, random_state=217).reset_index(drop=True)

# 11. Save the results in .csv file
#Original_RNA_Data_combined.to_csv('matthews_MFENFE2_training_data_ENTRNA_features.csv', index=False)










