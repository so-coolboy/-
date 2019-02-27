# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 11:49:38 2018

@author: www
"""

import pandas as pd
import numpy as np
import re


def t2h(t):
     if str(t) != '0':
          if ';' in t:
               t = t.replace(';',':')
          if '；' in t:
               t = t.replace('；',':')
          if '"' in t:
               t = t.replace('"',':')
          if '.' in t:
               t = t.replace('.',':')
          h=t.split(":")[0]
          m=t.split(":")[0]
          return (int(h)*3600+int(m)*60)/3600.0
     else:
          return -1

def getDuration(se):
     if str(se) != '0':
          if ';' in se:
               se = se.replace(';',':')
          if '；' in se:
               se = se.replace('；',':')
          if '::' in se:
               se = se.replace('::', ':')
          if '"' in se:
               se = se.replace('"',':')
          sh,sm,eh,em=re.split("[:,-]",se)
          if int(sh) > int(eh):
               return (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
          else:
               return (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
     else:
          return -1               
          
def get_sh(se):
     if str(se) != '0':
          sh = se.split('-')[0]
          return sh
     else:
          return 0

def get_eh(se):
     if str(se) != '0':
          eh = se.split('-')[1]
          return eh
     else:
          return 0          


train = pd.read_csv('../input/jinnan_round1_train_20181227.csv', encoding = 'gb18030')
testA = pd.read_csv('../input/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
testB = pd.read_csv('../input/jinnan_round1_testB_20190121.csv', encoding = 'gb18030')
subA = pd.read_csv('../input/jinnan_round1_submit_20181227.csv', encoding = 'gb18030')
testA = testB.copy()


#填充缺失值
#缺失值多的认为是可选工序，缺失值<=10的用中位数填充。
for col in ['A2','A7','A3','A8','B10','B11']:
     train[col] = train[col].fillna(0)
     testA[col] = testA[col].fillna(0)
for col in ['B1','A21','A23','B2','B3','B12','B13']:
     median = train[col].median()
     train[col] = train[col].fillna(median)
     testA[col] = testA[col].fillna(median)
   
     
train['A24'].fillna('3:00:00', inplace=True)
train['A26'].fillna('19:30:00', inplace=True)
train['B5'].fillna('14:00:00', inplace=True)
train['B8'].fillna(45, inplace=True)

testA['A20'].fillna('22:30:00-23:00:00', inplace=True)
testA['A25'].fillna(70, inplace=True)
testA['A27'].fillna(78, inplace=True)

#更改一些异常值
train['A5'].replace('1900/1/29 0:00','14:00:00',inplace=True)
train['A5'].replace('1900/1/21 0:00','20:30:00',inplace=True)
train['A9'].replace('1900/1/9 7:00','23:20:00', inplace=True)
train['A9'].replace('700','6:30:00', inplace=True)
train['A11'].replace(':30:00','0:30:00',inplace=True)
train['A11'].replace('1900/1/1 2:30','21:30:00',inplace=True)
train['A16'].replace('1900/1/12 0:00','12:00:00',inplace=True)
train['A25'].replace('1900/3/10 0:00',70, inplace=True)
train['A26'].replace('1900/3/13 0:00','13:00:00', inplace=True)
train['A20'].replace('6:00-6:30分', '6:00-6:30', inplace=True)
train['A20'].replace('6:30-7:0', '6:30-7:00', inplace=True)
train['A20'].replace('10:30-11:0', '10:30-11:00', inplace=True)
train['A20'].replace('10:00-11:0', '10:00-11:00', inplace=True)
train['A20'].replace('18:00:-18:30', '18:00-18:30', inplace=True)
train['A20'].replace('2:0-3:00', '2:00-2:30', inplace=True)
train['B4'].replace('15:00-1600','15:00-16:00', inplace=True) 
train['B4'].replace('14:00-15:0','14:00-15:00', inplace=True)
train['B4'].replace('15:0-16:30','15:00-16:30', inplace=True)
train['B4'].replace('16:00-17:002','16:00-17:00', inplace=True)
train['B4'].replace('17:0-18:00','17:00-18:00', inplace=True)
train['B4'].replace('19:-20:05','19:00-20:05', inplace=True)
train['B4'].replace('20:00-21:0','20:00-21:00', inplace=True)
train['B9'].replace('20:30-22:0','20:30-22:00', inplace=True)
train['B9'].replace('21:30-23:0','21:30-23:00', inplace=True)
train['B10'].replace('12:30-14:0','12:30-14:00', inplace=True)
train['B10'].replace('23:0-1:0','23:0-1:00', inplace=True)

train['A25']=train['A25'].astype('int')
testA['A5'].replace('1900/1/22 0:00','22:00:00',inplace=True)
testA['A20'].replace('14:0-14:30','14:00-14:30',inplace=True)
testA['A20'].replace('22:30:00-23:00:00','22:30-23:00',inplace=True)


#############################################################################
##############################做时间，温度，压强等的特征#########################
############################################################################
#时间列
time_col = ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']
#时间段列
ticu_col = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']
#温度列
tem_col = ['A6','A8','A10','A12','A15','A17','A21','A25','A27','B6','B8']
#原料列
mate_col = ['A1','A2','A3','A4','A19','B1','B12','B14']
#酸碱度列
ph_col = ['A22','A23','B2','B3','B13']
#压强列
pre_col = ['A13','A18']


#8，删除两个异常值
train = train[train['收率']>0.87]

 
data = pd.concat([train, testA])
categorical_columns = time_col + ticu_col 


#求时间列的特征     
#1, 各时间段间隔，转化为时做单位
dur_col = []
for i in ticu_col:
     data[i] = data[i].astype('str')
     data[i+'_Dur'] = data[i].apply(getDuration)
     dur_col.append(i+'_Dur')
     
     
#2,把时间段划分成两个时间列。
for i in ticu_col:
     data[i+'_sh'] = data[i].apply(get_sh)
     data[i+'_eh'] = data[i].apply(get_eh)
     categorical_columns.append(i+'_sh')
     categorical_columns.append(i+'_eh')
     
     
#3，时间转化，转化为时为单位
for i in data.columns:
     if (i in time_col) or ('sh' in i) or ('eh' in i):
          data[i] = data[i].astype('str')
          data[i+'copy'] = data[i].copy()
          data[i] = data[i].apply(t2h)
          categorical_columns.remove(i)
          categorical_columns.append(i+'copy')
                
          

#继续做时间转化，如果后一个时间小于前一个，说明过了一天，因此需要加24
def fun(a, b):
     if b>=a:
          return b
     else:
          while b<a:
               b=b+24
          return b
time_list = ['A5','A9','A11','A14','A16','A20_sh',
'A20_eh','A24','A26','A28_sh','A28_eh','B4_sh','B4_eh',
'B5','B7','B9_sh','B9_eh','B10_sh','B10_eh','B11_sh','B11_eh']      
for i in range(len(time_list)-1):
     data[time_list[i+1]] = data.apply(lambda row:fun(a=row[time_list[i]], b=row[time_list[i+1]]), axis=1) 
     
old_col = data.columns.tolist()            
#4，算上时间段，前一个时间与后一个时间之间的间隔     
for i in range(len(time_list)-1):
     data[time_list[i+1]+'-'+time_list[i]] = data[time_list[i+1]] - data[time_list[i]]
     
#隔两个的时间间隔
for i in range(len(time_list)-2):
     data[time_list[i+2]+'-'+time_list[i]] = data[time_list[i+2]] - data[time_list[i]]
          
#隔三个的时间间隔
for i in range(len(time_list)-3):
     data[time_list[i+3]+'-'+time_list[i]] = data[time_list[i+3]] - data[time_list[i]]

#隔四个的时间间隔
for i in range(len(time_list)-4):
     data[time_list[i+4]+'-'+time_list[i]] = data[time_list[i+4]] - data[time_list[i]]
          
##隔五个的时间间隔
for i in range(len(time_list)-5):
     data[time_list[i+5]+'-'+time_list[i]] = data[time_list[i+5]] - data[time_list[i]]


#5，不算时间段，前一个时间与后一个时间之间的间隔
time_list = ['A5','A9','A11','A14','A16','A24','A26','B5','B7']

for i in range(len(time_list)-1):
     data[time_list[i+1]+'-'+time_list[i]] = data[time_list[i+1]] - data[time_list[i]]
    
#隔两个的时间间隔
for i in range(len(time_list)-2):
     data[time_list[i+2]+'-'+time_list[i]] = data[time_list[i+2]] - data[time_list[i]]
    
#隔三个的时间间隔
for i in range(len(time_list)-3):
     data[time_list[i+3]+'-'+time_list[i]] = data[time_list[i+3]] - data[time_list[i]]
    
#隔四个的时间间隔
for i in range(len(time_list)-4):
     data[time_list[i+4]+'-'+time_list[i]] = data[time_list[i+4]] - data[time_list[i]]
         
##隔五个的时间间隔
for i in range(len(time_list)-5):
     data[time_list[i+5]+'-'+time_list[i]] = data[time_list[i+5]] - data[time_list[i]]

new_col = data.columns.tolist()
time_jiange_col = [i for i in new_col if i not in old_col]
old_col = new_col     
  
 
# 温度变化特征，温度差， 温度差除以对应时间差
tem_col = ['A6','A10','A12','A15','A17','A25','A27','B6','B8']
time_list = ['A5','A9','A11','A14','A16','A24','A26','B5','B7']
#隔一个的温度变化
for i in range(len(tem_col)-1):
     data[tem_col[i+1]+'-'+tem_col[i]] = data[tem_col[i+1]] - data[tem_col[i]]
     data[tem_col[i+1]+'-'+tem_col[i]+'/'+time_list[i+1]+'-'+time_list[i]] = data[tem_col[i+1]+'-'+tem_col[i]] / data[time_list[i+1]+'-'+time_list[i]]
   
#隔两个的温度变化
for i in range(len(tem_col)-2):
     data[tem_col[i+2]+'-'+tem_col[i]] = data[tem_col[i+2]] - data[tem_col[i]]
     data[tem_col[i+2]+'-'+tem_col[i]+'/'+time_list[i+2]+'-'+time_list[i]] = data[tem_col[i+2]+'-'+tem_col[i]] / data[time_list[i+2]+'-'+time_list[i]]
    
#隔三个的温度变化
for i in range(len(tem_col)-3):
     data[tem_col[i+3]+'-'+tem_col[i]] = data[tem_col[i+3]] - data[tem_col[i]]
     data[tem_col[i+3]+'-'+tem_col[i]+'/'+time_list[i+3]+'-'+time_list[i]] = data[tem_col[i+3]+'-'+tem_col[i]] / data[time_list[i+3]+'-'+time_list[i]]

#隔四个的温度变化
for i in range(len(tem_col)-4):
     data[tem_col[i+4]+'-'+tem_col[i]] = data[tem_col[i+4]] - data[tem_col[i]]
     data[tem_col[i+4]+'-'+tem_col[i]+'/'+time_list[i+4]+'-'+time_list[i]] = data[tem_col[i+4]+'-'+tem_col[i]] / data[time_list[i+4]+'-'+time_list[i]]

#隔五个的温度变化
for i in range(len(tem_col)-5):
     data[tem_col[i+5]+'-'+tem_col[i]] = data[tem_col[i+5]] - data[tem_col[i]]
     data[tem_col[i+5]+'-'+tem_col[i]+'/'+time_list[i+5]+'-'+time_list[i]] = data[tem_col[i+5]+'-'+tem_col[i]] / data[time_list[i+5]+'-'+time_list[i]]

new_col = data.columns.tolist()
wendu_jiange_col = [i for i in new_col if i not in old_col]
old_col = new_col  


#压强变化特征
data['A18-A13'] = data['A18'] - data['A13']
data['A18-A13/A16-A11'] = data['A18-A13'] / data['A16-A11']


#酸碱度列和浓度列
ph_col = ['A22','A23','B2','B3','B13']
data['A23-A22'] = data['A23'] - data['A22']
data['B1*B2'] = data['B1'] * data['B2']
data['B12*B13'] = data['B12'] * data['B13']
data['b14/A1_A3_A4_A19_B1_B12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])


#7，提取出id信息
data['id'] = data['样本id'].apply(lambda x: int(str(x).split('_')[1]))    
print(data.shape)


#删除测试集中为唯一列的数值
nun=[]
for  i in testA.columns:
     if testA[i].nunique()==1:
          nun.append(i)
          
data.drop(nun, axis=1, inplace=True)
print(data.shape)


#删除一些特征
good_cols = list(data.columns)
for col in data.columns:
    rate = data[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.95:
        good_cols.remove(col)
        if col in time_jiange_col:
             time_jiange_col.remove(col)
        if col in wendu_jiange_col:
             wendu_jiange_col.remove(col)
        print(col,rate)
        if col in categorical_columns:
             categorical_columns.remove(col)
data = data[good_cols]
print(data.shape)


#对类别列进行编码
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
#==============================================================================
# lb_columns = categorical_columns + tem_col + mate_col
# lb_columns.remove('A1')
# lb_columns.remove('A2')
# lb_columns.remove('A3')
# lb_columns.remove('A4')
#==============================================================================
lb_columns = categorical_columns 
for f in lb_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))

     
#data = pd.get_dummies(data, columns=categorical_columns) 
print(data.shape) 

train = data.loc[data['收率'].notnull()]  
testA = data.loc[data['收率'].isnull()]

                           
#==============================================================================
# train['intTarget'] = pd.cut(train['收率'], 5, labels=False)
# train = pd.get_dummies(train, columns=['intTarget'])
# li = ['intTarget_0','intTarget_1','intTarget_2','intTarget_3','intTarget_4']
# mean_columns = []
# for f1 in lb_columns:
#     cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
#     if cate_rate < 0.90:
#         for f2 in li:
#             col_name = 'B14_to_'+f1+"_"+f2+'_mean'
#             mean_columns.append(col_name)
#             order_label = train.groupby([f1])[f2].mean()
#             train[col_name] = train['B14'].map(order_label)
#             miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
#             if miss_rate > 0:
#                 train = train.drop([col_name], axis=1)
#                 mean_columns.remove(col_name)
#             else:
#                 testA[col_name] = testA['B14'].map(order_label)
#                 
# train.drop(li, axis=1, inplace=True)
#==============================================================================
#==============================================================================
# print(train.shape)
# print(testA.shape)                 
# 
# yuan_col = train.columns.tolist()                 
# 
#==============================================================================


###########################特征组合#################################################
#==============================================================================
# lb_columns = time_col + ticu_col + tem_col + mate_col
# lb_columns.remove('A1')
# lb_columns.remove('A2')
# lb_columns.remove('A3')
# lb_columns.remove('A4')
# lb_columns.remove('B14')
# #10，B14相对于其它列的聚合特征 
# for i in lb_columns:
#      label = train.groupby(i)['B14'].mean().reset_index()
#      label.columns = [i, i+'_B14_mean']
#      train = pd.merge(train, label, on=i, how='left')
#      testA = pd.merge(testA, label, on=i, how='left')
# #A6相对于其它列的聚合特征
# lb_columns.remove('A6')
# lb_columns.append('B14')
# for i in lb_columns:
#      label = train.groupby(i)['A6'].mean().reset_index()
#      label.columns = [i, i+'_A6_mean']
#      train = pd.merge(train, label, on=i, how='left')
#      testA = pd.merge(testA, label, on=i, how='left')
#      
#      
# #10，单个特征与组内平均值差和差的绝对值
# lb_columns.append('A6')
# col = []
# for i in yuan_col:
#      if i not in categorical_columns :
#           col.append(i)
# col.remove('样本id')
# col.remove('收率')
# data = pd.concat([train, testA])          
# for i in col:
#      mean = data[i].mean()
#      data[i+'_sub_mean'] = data[i] - mean
#      data[i+'_sub_mean_abs'] = data[i+'_sub_mean'].apply(lambda x: abs(x)) 
# 
#      
# #11，组合特征，这里求x+y，x-y，x*y，x/y 和 y/x
# col = dur_col
# for i in col:
#      for j in col:
#           if col.index(i)<col.index(j):
#                data[i+'_+_'+j] = data[i] + data[j]
#                data[i+'_*_'+j] = data[i] * data[j]
# 
# for i in col:
#      for j in col:
#           if i != j:
#                data[i+'_-_'+j] = data[i] - data[j]
#                data[i+'_/_'+j] = data[i] / data[j]
#        
# data.replace(np.inf, 999, inplace=True)
# data.replace(-np.inf, -999, inplace=True)
#           
# train = data.loc[data['收率'].notnull()]  
# testA = data.loc[data['收率'].isnull()]     
# 
# train.fillna(0, inplace=True)   
# testA.fillna(0, inplace=True)             
# 
# print(train.shape)
# print(testA.shape)
# 
#==============================================================================
############################################################################

       
target = train['收率']
del train['收率']
del testA['收率']
del train['样本id']
del testA['样本id']



#==============================================================================
#####################################################################################
#                        以下进行特征选择                                              #
#####################################################################################
######################方法一
def modeling_cross_validation(params, X, y, nr_folds=5):
    
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    folds = KFold(n_splits=nr_folds, shuffle=False, random_state=4096)
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("fold n°{}".format(fold_+1))
        trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
        val_data = lgb.Dataset(X[val_idx], y[val_idx])

        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)
        oof_preds[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)

    score = mean_squared_error(oof_preds, target)
    
    return  score/2
    
    
def featureSelect(init_cols):
    params = {'num_leaves': 120,
             'min_data_in_leaf': 30, 
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.05,
             "min_child_samples": 30,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.02,
             "verbosity": -1}
    best_cols = init_cols.copy()
    best_score = modeling_cross_validation(params, train[init_cols].values, target.values, nr_folds=5)
    print("初始CV score: {:<8.8f}".format(best_score))
    for f in init_cols:

        best_cols.remove(f)
        score = modeling_cross_validation(params, train[best_cols].values, target.values, nr_folds=5)
        diff = best_score - score
        print('-'*10)
        if diff > 0.0000002:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 有效果,删除！！".format(f,score,best_score))
            best_score = score
        else:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 没效果,保留！！".format(f,score,best_score))
            best_cols.append(f)
    print('-'*10)
    print("优化后CV score: {:<8.8f}".format(best_score))
    
    return best_cols
    
best_features = featureSelect(train.columns.tolist())
print(best_features)    


train = train[best_features]
testA = testA[best_features]


#==============================================================================
# train.to_csv('../feature/train_my.csv', index=False)
# testA.to_csv('../feature/testA_my.csv', index=False)
#==============================================================================



#####################################################################################
#                        以上进行特征选择                                              #
#####################################################################################

y_train = target.values
test = testA.values
X_train = train.values


import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
param = {'num_leaves': 120,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
         
# 五折交叉验证
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, 
                    trn_data, 
                    num_round, 
                    valid_sets = [trn_data, val_data], 
                    verbose_eval = 200, 
                    early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    
    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
     
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = train.columns
    fold_importance["importance"] = clf.feature_importance()
    fold_importance["fold"] = fold_ + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

print("CV score: {:<8.7f}".format(mean_squared_error(oof, target)))
feature_col = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)


#==============================================================================
# sub_df = pd.read_csv('../input/jinnan_round1_submit_20181227.csv', header=None)
# sub_df[1] = predictions
# sub_df.to_csv("../sub/sub.csv", index=False, header=None)
#==============================================================================


sub = pd.DataFrame()
sub['id'] = testA['样本id']
sub['pr'] = predictions
sub.to_csv("../sub/sub.csv", index=False, header=None)
    
     
     
     