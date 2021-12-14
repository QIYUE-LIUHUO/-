import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

data = pd.read_csv('train .csv')  # 读训练集
# data.columns
# print(data)   # 查看训练集
data.info()  # 查看数据集概要信息
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
"""
survived：乘客的生还情况（1活，0死）
Pclass：乘客的社会阶层 (1上层，2中层，3底层)
sex：乘客的年龄
age：乘客的年纪
SibSp： 一起上船的兄弟姐妹和配偶个数
Parch： 一起上船的父母和子女个数
Fare：乘客支付的票价
cabin:仓位号码
embarked:登船的港口
"""
data.isnull().sum()  # 可以查看缺失的训练级
data['Age'] = data['Age'].fillna(data['Age'].mean())  # 求取年龄的平均值，用于fillna填充缺失值
data['Cabin'] = pd.factorize(data.Cabin)[0]  # factorize函数可以将Series中的标称型数据映射称为一组数字，相同的标称型映射为相同的数字
data.fillna(0, inplace=True)  # 剩余的缺失值用0填充
data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]  # 男为1女为0

#  船舱堵塞等级变为散列
data['p1'] = np.array(data['Pclass'] == 1).astype(np.int32)
data['p2'] = np.array(data['Pclass'] == 2).astype(np.int32)
data['p3'] = np.array(data['Pclass'] == 3).astype(np.int32)

del data['Pclass']  # 删除此列

# 将登船的港口也变为三列
data.Embarked.unique()  # 将港口内重复的元素去掉并将元素有大到小返回
data['e1'] = np.array(data['Embarked'] == 'S').astype(np.int32)
data['e2'] = np.array(data['Embarked'] == 'C').astype(np.int32)
data['e3'] = np.array(data['Embarked'] == 'Q').astype(np.int32)

del data['Embarked']  # 删除此列

# data.values.dtype

# 数据集最后拥有的数据分类
data_train = data[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'p1', 'p2', 'p3', 'e1', 'e2', 'e3']].values

data_target = data['Survived'].values.reshape(len(data), 1)
np.shape(data_train), np.shape(data_target)

# 搭建训练网络
x = tf.placeholder("float", shape=[None, 12])
y = tf.placeholder("float", shape=[None, 1])

# 前向传播过程
weight = tf.Variable(tf.random.normal([12, 1]))
bias = tf.Variable(tf.random.normal([1]))
output = tf.matmul(x, weight) + bias
pred = tf.cast(tf.sigmoid(output) > 0.5, tf.float32)  # 预测结果大于0.5值设为1，否则为0
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))

# 梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.0003).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

#  测试数据预处理
data_test = pd.read_csv('test .csv')
data_test = data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
data_test['Age'] = data_test['Age'].fillna(data_test['Age'].mean())
data_test['Cabin'] = pd.factorize(data_test.Cabin)[0]  # 数值化·
data_test.fillna(0, inplace=True)  # 剩余的缺失值用0填充

data_test['Sex'] = [1 if x == 'male' else 0 for x in data_test.Sex]  # 同上

# 这两个的数据就是测试的p和e的处理方式同上
data_test['p1'] = np.array(data_test['Pclass'] == 1).astype(np.int32)
data_test['p2'] = np.array(data_test['Pclass'] == 2).astype(np.int32)
data_test['p3'] = np.array(data_test['Pclass'] == 3).astype(np.int32)
data_test['e1'] = np.array(data_test['Embarked'] == 'S').astype(np.int32)
data_test['e2'] = np.array(data_test['Embarked'] == 'C').astype(np.int32)
data_test['e3'] = np.array(data_test['Embarked'] == 'Q').astype(np.int32)
del data_test['Pclass']
del data_test['Embarked']

test_lable = pd.read_csv(r'titanic_泰坦尼克数据集\gender_submission.csv')
test_lable = np.reshape(test_lable.Survived.values.astype(np.float32), (418, 1))

st = time.time()
# 开始训练
sess = tf.Session()  # 开启
sess.run(tf.global_variables_initializer())  # 将所有的变量初始化
loss_train = []
train_acc = []
test_acc = []

name = input("这是第几次跑数据")
#  方便观看写入csv文件
fp = open(name+"泰坦尼克.csv", "a+", newline="")
# 修饰，处理成支持scv读取的文件
csv_fp = csv.writer(fp)
# 设置csv文件内标题头
head = ['loss_temp', 'train_acc_temp', 'test_acc_temp']
# 写入标题
csv_fp.writerow(head)
data = []  # 用于存放跑出来的数据


for i in range(25000):
    one_data = []  # 用于存放一次的数据
    index = np.random.permutation(len(data_target))  # 将顺序打乱，避免出现过拟合
    data_train = data_train[index]
    data_target = data_target[index]
    for n in range(len(data_target) // 100 + 1):
        batch_xs = data_train[n * 100:n * 100 + 100]
        batch_ys = data_target[n * 100:n * 100 + 100]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 1000 == 0:
        loss_temp = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
        loss_train.append(loss_temp)
        data = [loss_temp]  # 读取数据
        if data: # 如果有数据
            one_data.append(data[0]) # 存放
        train_acc_temp = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
        train_acc.append(train_acc_temp)
        data = [train_acc_temp] # 同上
        if data:  # 如果有数据
            one_data.append(data[0])
        test_acc_temp = sess.run(accuracy, feed_dict={x: data_test, y: test_lable})
        test_acc.append(test_acc_temp)
        data = [test_acc_temp] # 同上
        if data:  # 如果有数据
            one_data.append(data[0])
        if len(one_data) == 3: # 三个数据为一组，如果数据为一组就存放
            data.append(one_data)
            csv_fp.writerow(one_data)
        print(loss_temp, train_acc_temp, test_acc_temp)

fp.close()  # 关闭文件




ed = time.time()
print("用时：", ed - st)
# 画图  做出数据可视化D
plt.plot(loss_train, 'k-')
plt.title('train loss')
plt.show()
plt.plot(train_acc, 'b-', label='train_acc')
plt.plot(test_acc, 'r--', label='train_acc')
plt.title('train and test accuracy')
plt.legend()
plt.show()
