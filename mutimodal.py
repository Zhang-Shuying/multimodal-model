# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:26:14 2023

@author: LENOVO
"""




#multimodal model

import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from tensorflow.keras.models import load_model


#Import Data
train = pd.read_csv('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/data/train.csv')
test = pd.read_csv('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/data/test.csv')
vad = pd.read_csv('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/data/vad.csv')


y_train = train[['logPtb']]
y_test = test[['logPtb']]
y_vad = vad[['logPtb']]



x_train = train.iloc[:,6:]
x_test = test.iloc[:,6:]
x_vad = vad.iloc[:,6:]



#load model
model = load_model('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/muti-pub1.h5')


fit = model.predict(x_train)
pre = model.predict(x_test)
pvad = model.predict(x_vad)
r2_tra = r2_score(y_train,fit)
r2_ext = r2_score(y_test, pre)
r2_vad = r2_score(y_vad, pvad)

rmse_tra = np.sqrt(mean_squared_error(y_train,fit))
rmse_ext = np.sqrt(mean_squared_error(y_test,pre))
rmse_vad = np.sqrt(mean_squared_error(y_vad,pvad))

mae_tra = mean_absolute_error(y_train,fit)
mae_ext = mean_absolute_error(y_test,pre)
mae_vad = mean_absolute_error(y_vad,pvad)


print('R2_tra:',r2_tra) 
print('RMSE_tra:',rmse_tra) 
print('MAE_tra:', mae_tra)              
print('R2_ext:',r2_ext)                                 
print('RMSE_ext:',rmse_ext)    
print('MAE_ext:',mae_ext)  
print('R2_vad:',r2_vad)                                 
print('RMSE_vad:',rmse_vad)    
print('MAE_vad:',mae_vad)  






#SHAP analysis
import sys
os.environ['CONDA_PREFIX'] = 'D:\\ProgramData\\Anaconda3\\envs\\myenv'
sys.path.insert(0, os.environ['CONDA_PREFIX'])
import shap
import numpy as np


explainer = shap.Explainer(model, x_vad)
shap_values = explainer(x_vad, max_evals=2 * x_vad.shape[1] + 1)



import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 80
shap.summary_plot(shap_values, x_vad, show=False)


#calculate shap values
feature_names = x_vad.columns.tolist() 

absolute_mean_shap_values = np.abs(shap_values.values).mean(axis=0)
top_features_indices = np.argsort(absolute_mean_shap_values)[::-1][:20]
top_20_shap_values = absolute_mean_shap_values[top_features_indices]

print("Top 20 features mean absolute SHAP values:")
for i, index in enumerate(top_features_indices):
    feature_name = feature_names[index]
    shap_value = top_20_shap_values[i]
    print(f"{feature_name}: {shap_value}")
    
    
    




#AD characterization
import os
os.chdir('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision')

from AppDomain import NSG, NSGVisualizer

df_train = pd.read_csv('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/data/train_all-AD.csv')
df_vad = pd.read_csv('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/data/vad-AD.csv')

        
df_train.set_index(['species_idx', 'tissue_idx', 'path_idx', 'phase_idx'], inplace=True)
df_vad.set_index(['species_idx', 'tissue_idx', 'path_idx', 'phase_idx'], inplace=True)


common_index = df_vad.index.intersection(df_train.index)


vad_data_filtered = df_vad.loc[common_index]   
vad_data_filtered.reset_index(inplace=True)

vad = vad_data_filtered
y_vad = vad ['logPtb']
x_vad = vad.drop(columns=['CmpdID', 'CAS', 'SMILES', 'logPtb','species','tissue','path','phase'])
pvad = model.predict(x_vad)
pvad = pd.DataFrame(pvad)


r2_vad = r2_score(y_vad, pvad)
rmse_vad = np.sqrt(mean_squared_error(y_vad,pvad))
mae_vad = mean_absolute_error(y_vad,pvad)


print('R2_vad:',r2_vad)                                 
print('RMSE_vad:',rmse_vad)    
print('MAE_vad:',mae_vad)  









P_list =[]
I_list =[]
r2_vad_list =[]
rmse_vad_list =[]
mae_vad_list =[]
num_remained_list = []


for P in [0.001,0.01,0.1,1,5,10]:
    for I in [0.3,0.5,0.7,0.9,1,1.2]:
        df_train = pd.read_csv('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/data/train_all-AD1.csv')
        df_vad = pd.read_csv('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/data/vad_data_filtered.csv')

       
        train_CmpdID_list = df_train.CmpdID.values
        vad_CmpdID_list = df_vad.CmpdID.values
                 
        nsg = NSG(df_train, smiCol='SMILES', yCol='logPtb')
        nsg.calcPairwiseSimilarityWithFp('MACCS_keys')  
        QTSM = nsg.genQTSM(df_vad, smiCol='SMILES')   
        ADMetric = nsg.queryADMetrics(QTSM)
        ADMetric['CmpdID']=vad_CmpdID_list
        
        Pscore_ls = ADMetric.simiDensity.values
        Iscore_ls = ADMetric.simiWtLD_w.values
        
        CmpdID_ls = vad_CmpdID_list
        CmpdinAD = []
        
        
        for i in range(len(Pscore_ls)):
            if Pscore_ls[i] > P and Iscore_ls[i] < I:  
                CmpdinAD.append(CmpdID_ls[i])
            else:
                continue

        

        vad = vad_data_filtered[vad_data_filtered['CmpdID'].isin(CmpdinAD)]    
        y_vad = vad ['logPtb']
        x_vad = vad.drop(columns=['CmpdID', 'CAS', 'SMILES', 'logPtb','species','tissue','path','phase'])




        pvad = model.predict(x_vad)
        pvad = pd.DataFrame(pvad)

       

        r2_vad = r2_score(y_vad, pvad)
        rmse_vad = np.sqrt(mean_squared_error(y_vad,pvad))
        mae_vad = mean_absolute_error(y_vad,pvad)



        P_list.append(P)
        I_list.append(I)
        r2_vad_list.append(r2_vad)
        rmse_vad_list.append(rmse_vad)
        mae_vad_list.append(mae_vad)
        num_remained_list.append(len(vad))    
        
        print(str(P)
              +"\n"+str(I)
              +"\n"+str(r2_vad)
              +"\n"+str(rmse_vad)
              +"\n"+str(mae_vad)
              +"\n"+str(len(vad))
             )
       
all_list = {'P': P_list
           ,'I': I_list
           ,'r2_ext': r2_vad_list
           ,'rmse_ext': rmse_vad_list
           ,'mae_ext': mae_vad_list
           ,'num_remained': num_remained_list
}
all_list
all_list_df = pd.DataFrame(all_list)

all_list_df.to_csv('D:/学习文献/课题/pp-lfer/多任务论文/mammal and fish/revision/data/AD_ANN_a_20.csv')
