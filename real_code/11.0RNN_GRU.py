import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv('E:/2021-5-13下载/stock_data/stock_data/sh300index.csv',index_col=0)
valid_set_size_percentage = 10
test_set_size_percentage = 10

def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
    return df

def load_data(stock, seq_len):
    data_raw = stock.to_numpy() # pd to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]

    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]

    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# 去除冗余指标
df_stock = df.copy()
df_stock.drop(['vol'],1,inplace=True)
df_stock.drop(['lastclose'],1,inplace=True)
df_stock.drop(['label'],1,inplace=True)
df_stock.drop(['ZTM:ma5'],1,inplace=True)
df_stock.drop(['ZTM:ma7'],1,inplace=True)
df_stock.drop(['ZTM:ma10'],1,inplace=True)
df_stock.drop(['ZTM:ma21'],1,inplace=True)
df_stock.drop(['holdingvol'],1,inplace=True)
df_stock.drop(['ZTM:MACD'],1,inplace=True)
df_stock.drop(['ZTM:RSI'],1,inplace=True)

cols = list(df_stock.columns.values)
df_stock_norm = normalize_data(df_stock)
seq_len = 20
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)
# 对训练数据随机化处理
index_in_epoch = 0;
perm_array = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)  # shuffle permutation array
        start = 0  # start next epoch
        index_in_epoch = batch_size

    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

# 定义超参
n_steps = seq_len - 1
# 输入大小（与指标数量对应）
n_inputs = 4
n_neurons = 200
# 输出大小（与指标数量对应）
n_outputs = 4
# 层数
n_layers = 2
# 学习率
learning_rate = 0.001
# 批大小
batch_size = 50

# 迭代训练次数
n_epochs = 20
# 训练集大小
train_set_size = x_train.shape[0]
# 测试集大小
test_set_size = x_test.shape[0]
n_batches = train_set_size//batch_size

layers = [tf.keras.layers.GRUCell(units=n_neurons,activation=tf.nn.leaky_relu) for layer in range(n_layers)]
rnn_layer = tf.keras.layers.RNN(cell=layers,return_sequences=True)
input_tensor = tf.keras.layers.Input([n_steps, n_inputs])#50,19,4
rnn_outputs = rnn_layer(input_tensor)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.keras.layers.Dense(n_outputs)(stacked_rnn_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:, n_steps - 1, :]  # 定义输出
model = tf.keras.Model(input_tensor,outputs)

class MeanSquareLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred-y_true))  # 使用MSE作为损失

def compute_loss(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))  # 使用MSE作为损失

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
global_steps = 0
def train_step(x,y):
    global global_steps
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = compute_loss(y,y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP {}    epoch:{}      index:{}     loss:{}    ".format(global_steps,epoch,index_in_epoch,loss))
        global_steps += 1
for epoch in range(n_epochs):
    for i in range(n_batches):
        x , y = get_next_batch(batch_size)
        train_step(x,y)


ft = 0 # 0 = open, 1 = close, 2 = highest, 3 = lowest

y_train_pred = model(x_train)
y_test_pred = model(x_test)
y_valid_pred = model(x_valid)
#结果可视化
plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);

plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                   y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
         label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
         y_valid_pred[:,ft], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                   y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')

plt.subplot(1,2,2)

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')




# model.compile(optimizer=optimizer,loss=MeanSquareLoss)
# model.fit(x = x_train,y = y_train,batch_size=batch_size,epochs=n_epochs,validation_data=(x_test,y_test))


# training_op = optimizer.minimize(loss)