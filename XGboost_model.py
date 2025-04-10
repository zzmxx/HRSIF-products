# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 07:28:36 2025

@author: zzmxx
"""

##### train XGBoost to predict SIF
####

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os
import xgboost as xgb
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import time


from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#%%

#将所有数据合并为一个dataframe
file_dir = r'...\data'
files = os.listdir(file_dir)
df1 = pd.read_csv(os.path.join(file_dir, files[0]))#,dtype='float32'

for e in files[1:]:
    df2 = pd.read_csv(os.path.join(file_dir, e))#,dtype='float32'
    df1 = pd.concat((df1, df2), axis=0, join='inner')
print(df1)
data = df1
# Dividing features and target
#X = data.drop(columns = ['SIF'], axis = 1)
X = data[['Red','NIR','Blue','Green','IR1','IR2','IR3','NDVI','LST','PAR_CERES','IGBP']]
y = data[['SIF']]
#%%
# Categorical variables
category_feature_mask = X.dtypes == object
category_cols = X.columns[category_feature_mask].tolist()
X_categorical = X[category_cols].copy()
X_continuous = X.drop(category_cols, axis = 1)

# Continuous variables normalization
std = StandardScaler()
X_continuous_data = std.fit_transform(X_continuous)
X_continuous_df = pd.DataFrame(X_continuous_data, columns = X_continuous.columns)
# OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_categorical)
known_categories = X_categorical.iloc[:,0].unique().tolist()
X_categorical_data = enc.transform(X_categorical).toarray()

X_categorical_df = pd.DataFrame(X_categorical_data, columns = [enc.get_feature_names_out()])
X_new = pd.concat([X_continuous_df, X_categorical_df],axis=1)

# Split training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, shuffle = True, random_state=1729)


#%%
# n_estimators
cv_params = {'n_estimators': [3000, 3500, 4000, 4500, 5000]}  #1000, 1500, 2000, 2500, 3000, 3500, 4000 3000, 3500, 4000, 4500, 5000
other_params = {'learning_rate': 0.1, 'n_estimators': 1500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method' : 'hist'}#gpu_hist改成hist
#fit_params = {'early_stopping_rounds': 50}#已经没有这个参数了
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=2, verbose=1, n_jobs=4)#, fit_params=fit_params)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
print('each_iteration:{0}'.format(evalute_result))
print('best_params：{0}'.format(optimized_GBM.best_params_))
print('best_score:{0}'.format(optimized_GBM.best_score_))

# min_child_weight and max_depth
cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {'learning_rate': 0.07, 'n_estimators': 3500, 'max_depth': 1, 'min_child_weight': 1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method' : 'hist'}#,'device':"cuda"}
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=2, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
print('each_iteration:{0}'.format(evalute_result))
print('best_params：{0}'.format(optimized_GBM.best_params_))
print('best_score:{0}'.format(optimized_GBM.best_score_))

# learning_rate
cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 3500, 'max_depth': 7, 'min_child_weight':1, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method' : 'hist'}
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=2, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
print('each_iteration:{0}'.format(evalute_result))
print('best_params：{0}'.format(optimized_GBM.best_params_))
print('best_score:{0}'.format(optimized_GBM.best_score_))

#%% final model

# 定义模型的eval_set，也是后期绘制模型学习曲线的必须设置的参数
evalset = [(X_train, y_train), (X_test, y_test)]
# 定义模型
XGBR = xgb.XGBRegressor(n_estimators=3500,
                        max_depth=7, 
                        min_child_weight=1,
                        learning_rate=0.05,
                        tree_method='hist')    #2000,6,2,0.05  1500,7,1,0.05

X_train_pred = XGBR.predict(X_train)
start = time.perf_counter() 
y_pred = XGBR.predict(X_test)
elapsed = (time.perf_counter() - start)
r2 = r2_score(y_test, y_pred)
print('Accuracy: %.3f' % r2)

results = XGBR.evals_result()
plt.plot(results['validation_0']['rmse'], color='r',label='train')
plt.plot(results['validation_1']['rmse'], color="g",label='test')
plt.title('RMSE')
#
plt.legend()
plt.show()
#输出训练集评价指标
print("Time used:", elapsed)
print(f'RMSE : {np.sqrt(mean_squared_error(y_train, X_train_pred))}')
print(f'MAE : {mean_absolute_error(y_train, X_train_pred)}')
print(f'R2 : {r2_score(y_train, X_train_pred)}')

#%% 分析模型在不同IGBP类型下的表现
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 在导入模块后立即设置全局字体
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 如果需要显示中文，可以添加以下设置
plt.rcParams['axes.unicode_minus'] = False 

# 首先需要获取原始的IGBP值和对应的预测结果
# 由于我们的特征已经经过了OneHotEncoder编码，需要先恢复原始的IGBP值

# 获取测试集的原始数据
# 假设我们有原始的X_test_original（包含原始IGBP值的测试集）
# 如果没有保存，可以从原始数据中重新获取
# 这里我们从原始数据中提取对应测试集的记录

# 获取测试集的索引
test_indices = []
if hasattr(X_test, 'index'):
    test_indices = X_test.index.tolist()
else:
    # 如果测试集没有保留原始索引，则需要重新划分一次数据集来获取索引
    # 注意：这里使用相同的随机种子确保划分结果一致
    _, _, _, _, test_indices = train_test_split(X_new, y, test_size=0.2, 
                                               shuffle=True, random_state=1729, 
                                               return_indices=True)

# 从原始数据中提取测试集的IGBP值
original_X = data[['Red','NIR','Blue','Green','IR1','IR2','IR3','NDVI','LST','PAR_CERES','IGBP']]
original_y = data[['SIF']]
X_test_original = original_X.iloc[test_indices]
y_test_original = original_y.iloc[test_indices]

# 获取测试集的IGBP值
igbp_values = X_test_original['IGBP'].values

# 如果IGBP是数值型的，可能需要将其转换为字符串类型
if not isinstance(igbp_values[0], str):
    igbp_values = [str(int(x)) for x in igbp_values]

# 创建一个DataFrame来存储测试集的IGBP值和预测结果
results_df = pd.DataFrame({
    'IGBP': igbp_values,
    'True': y_test.values.flatten(),
    'Predicted': y_pred
})

# 按IGBP分组计算评价指标
igbp_metrics = {}
unique_igbp = np.unique(igbp_values)

for igbp in unique_igbp:
    group = results_df[results_df['IGBP'] == igbp]
    if len(group) > 10:  # 确保有足够的样本进行评估
        true_values = group['True']
        pred_values = group['Predicted']
        
        r2 = r2_score(true_values, pred_values)
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        mae = mean_absolute_error(true_values, pred_values)
        
        igbp_metrics[igbp] = {
            'count': len(group),
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        }

# 将结果转换为DataFrame以便于展示
metrics_df = pd.DataFrame.from_dict(igbp_metrics, orient='index')
metrics_df = metrics_df.sort_values('count', ascending=False)
print(metrics_df)

# 可视化不同IGBP类型的模型表现
plt.figure(figsize=(14, 8))

# 绘制R2柱状图
plt.subplot(1, 3, 1)
plt.bar(metrics_df.index, metrics_df['R2'], color='skyblue')
plt.xlabel('IGBP type')
plt.ylabel('R² scores')
plt.title('R² scores of IGBP')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 绘制RMSE柱状图
plt.subplot(1, 3, 2)
plt.bar(metrics_df.index, metrics_df['RMSE'], color='salmon')
plt.xlabel('IGBP type')
plt.ylabel('RMSE')
plt.title('RMSE scores of IGBP')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 绘制样本数量柱状图
plt.subplot(1, 3, 3)
plt.bar(metrics_df.index, metrics_df['count'], color='lightgreen')
plt.xlabel('IGBP type')
plt.ylabel('numbers of samples')
plt.title('numbers of samples of IGBP')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(r'F:\tropomisif\IGBP_performance.png', dpi=300)
plt.show()

# 绘制每个IGBP类型的真实值vs预测值散点图
plt.figure(figsize=(15, 10))
n_cols = 3
n_rows = (len(unique_igbp) + n_cols - 1) // n_cols

for i, igbp in enumerate(unique_igbp):
    group = results_df[results_df['IGBP'] == igbp]
    if len(group) > 10:  # 确保有足够的样本
        plt.subplot(n_rows, n_cols, i+1)
        plt.scatter(group['True'], group['Predicted'], alpha=0.5)
        
        # 添加对角线（理想预测线）
        max_val = max(group['True'].max(), group['Predicted'].max())
        min_val = min(group['True'].min(), group['Predicted'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 添加回归线
        z = np.polyfit(group['True'], group['Predicted'], 1)
        p = np.poly1d(z)
        plt.plot(group['True'], p(group['True']), 'g-')
        
        plt.title(f'IGBP={igbp}, R²={igbp_metrics[igbp]["R2"]:.3f}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'F:\tropomisif\IGBP_scatter_plots.png', dpi=300)
plt.show()

# 创建一个热力图来展示不同IGBP类型的各项指标
plt.figure(figsize=(12, 8))
metrics_heatmap = metrics_df[['R2', 'RMSE', 'MAE']].copy()
sns.heatmap(metrics_heatmap, annot=True, cmap='coolwarm', fmt='.3f', linewidths=.5)
plt.title('不同IGBP类型的模型性能指标')
plt.savefig(r'F:\tropomisif\IGBP_metrics_heatmap.png', dpi=300)
plt.show()

# 将结果保存到CSV文件
metrics_df.to_csv(r'F:\tropomisif\IGBP_performance_metrics.csv')

#%%保存训练模型
import joblib
joblib.dump(XGBR,r'E:/...')
