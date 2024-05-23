import numpy as np
from sklearn import linear_model, metrics
import os
filelist = os.listdir(os.getcwd())
paths = []
for file in filelist:
    if 'seed' in file:
        paths.append(file)
lags = [0,1,2,3,4,5,6]


def concat_x_byx(x, y, lag):
    x_fuse = [] 
    x_fuse.append(x)
    for i in range(lag):
        i = i + 1
        x_tmp = np.zeros_like(x)
        x_tmp[:, i:, :] = x[:, :-i, :]
        x_fuse.append(x_tmp)

    x_side = np.concatenate(x_fuse, axis = -1)
    return x_side.reshape(-1, x_side.shape[-1]), y.reshape(-1, y.shape[-1])

def concat_x_byy(x, y, lag):
    x_fuse = [] 
    x_fuse.append(x)
    for i in range(lag):
        i = i + 1
        x_tmp = np.zeros_like(y)
        x_tmp[:, i:, :] = y[:, :-i, :]
        x_fuse.append(x_tmp)

    x_side = np.concatenate(x_fuse, axis = -1)
    return x_side.reshape(-1, x_side.shape[-1]), y.reshape(-1, y.shape[-1])

def check_lag_byx(xt, yt, lag):   
    x,y = concat_x_byx(xt, yt, lag)
    train_num = int(0.5 * x.shape[0])
    x_train, x_test = x[train_num:], x[:train_num]
    y_train, y_test = y[train_num:], y[:train_num]
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    idxs = range(0, 200000, 10)
    y_test[idxs] = 1
    y_pred[idxs] = 1
    return metrics.mean_squared_error(y_test, y_pred)

def check_lag_byy(xt, yt, lag):   
    x,y = concat_x_byy(xt, yt, lag)
    train_num = int(0.5 * x.shape[0])
    x_train, x_test = x[train_num:], x[:train_num]
    y_train, y_test = y[train_num:], y[:train_num]
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    idxs = range(0, 200000, 10)
    y_test[idxs] = 1
    y_pred[idxs] = 1
    return metrics.mean_squared_error(y_test, y_pred)

res = {}

for path in paths:
    data = np.load(path + "/data.npz")
    xt = data['xt']
    yt = data['yt']
    for i in lags:
        ans = check_lag_byx(xt,yt,i) - check_lag_byx(xt,yt,0) 
        # print(path, ans)
        res[(path,i)] = ans
# res = sorted(res.items(), key = lambda kv:(kv[1], kv[0]))

for x in res:
    print(x, res[x])