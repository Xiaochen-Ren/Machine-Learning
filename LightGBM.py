import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

'''----------load 数据集-----------'''
dataset = datasets.load_iris()
x = dataset.data
target = dataset.target
#y = np.reshape(target,(len(target), 1))
y = dataset.target

x_train, x_verify, y_train, y_verify = train_test_split(x, y, random_state=1)
print('原始数据特征：', x.shape ,
      '，训练数据特征：', x_train.shape , 
      '，测试数据特征：',x_verify.shape )
print('原始数据标签',y.shape,
     '训练数据标签',y_train.shape,
     '测试数据标签',y_verify.shape)

'''----------定义线性回归模型，进行训练、预测-----------'''
import lightgbm as lgb
lgb_train = lgb.Dataset(x_train,y_train)# 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(x_train, y_train, reference=lgb_train)# 创建验证数据
param = {
    'task': 'train',
    'num_class':3,
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'multiclass', # 目标函数
    'metric': 'multi_logloss',  # 评估函数
    'num_leaves': 31,   # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
num_round = 10
gbm = lgb.train(param, lgb_train, num_round, valid_sets=[lgb_eval])

'lr.score(x_verify,y_verify)'
y_pred = list(gbm.predict(x_verify))
y_pred_copy = []
for y_ in y_pred:
    y_pred_copy.append(list(y_).index(max(y_)))

'''----------图形化预测结果-----------'''
plt.plot( range(len(y_verify)), y_verify, 'r', label='y_verify')
plt.plot( range(len(y_pred_copy)), y_pred_copy, 'g--', label='y_predict' )
plt.title('sklearn: Linear Regression')
plt.legend()
plt.show()

"""
 
'''----------输出模型参数、评价模型-----------'''
print("MSE:",metrics.mean_squared_error(y_verify,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_verify,y_pred)))
 
#输出模型对应R-Square
print("训练集得分:",gbm.score(x_train,y_train))
print("测试集得分:",gbm.score(x_verify,y_verify))
"""
