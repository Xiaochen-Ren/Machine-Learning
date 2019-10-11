import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

'''----------load 数据集-----------'''
dataset = datasets.load_iris()
x = dataset.data
target = dataset.target
y = np.reshape(target,(len(target), 1))

x_train, x_verify, y_train, y_verify = train_test_split(x, y, random_state=1)
print('原始数据特征：', x.shape ,
      '，训练数据特征：', x_train.shape , 
      '，测试数据特征：',x_verify.shape )
print('原始数据标签',y.shape,
     '训练数据标签',y_train.shape,
     '测试数据标签',y_verify.shape)

'''----------定义线性回归模型，进行训练、预测-----------'''

from sklearn.svm import SVC 
lr = SVC(kernel='linear',C=1E10)#支持向量机
#from sklearn.naive_bayes import GaussianNB
#lr = GaussianNB()#朴素贝叶斯
lr.fit(x_train,y_train)
lr.score(x_verify,y_verify)
y_pred = lr.predict(x_verify)

'''----------图形化预测结果-----------'''
plt.plot( range(len(y_verify)), y_verify, 'r', label='y_verify')
plt.plot( range(len(y_pred)), y_pred, 'g--', label='y_predict' )
plt.title('sklearn: Linear Regression')
plt.legend()
plt.show()


 
'''----------输出模型参数、评价模型-----------'''
print(lr.coef_)
print(lr.intercept_)
print("MSE:",metrics.mean_squared_error(y_verify,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_verify,y_pred)))
 
#输出模型对应R-Square
print("训练集得分:",lr.score(x_train,y_train))
print("测试集得分:",lr.score(x_verify,y_verify))
