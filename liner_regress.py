from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取
from sklearn.linear_model import Ridge,RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
'''----------load 数据集-----------'''
dataset = datasets.load_boston()
'''
 x 训练特征：['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
'''
x = dataset.data
 
target = dataset.target
#把label变为(?, 1)维度，为了使用下面的数据集合分割
y = np.reshape(target,(len(target), 1))
 
#讲数据集1:3比例分割为 测试集：训练集
x_train, x_verify, y_train, y_verify = train_test_split(x, y, random_state=1)
 
'''
x_train的shape：(379, 13)
y_train的shape：(379, 1)
x_verify的shape：(127, 13)
y_verify 的shape：(127, 1)
'''
 
 
'''----------定义线性回归模型，进行训练、预测-----------'''
lr = linear_model.LinearRegression()
# lr = Lasso(alpha=0.01) # Lasso回归
# lr = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
# lr = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
# lr = Ridge(alpha=0.5)
# lr = RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
lr.fit(x_train,y_train)
y_pred = lr.predict(x_verify)
 
 
'''----------图形化预测结果-----------'''
#只显示前50个预测结果，太多的话看起来不直观
plt.xlim([0,50])
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
