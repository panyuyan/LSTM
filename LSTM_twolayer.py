# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_test_split_1dim(normalize_data,time_step,intervals): #using minmax to normalize
    data_x,data_y = [],[]    #first column is t, second column is t+1
    for i in range(len(normalize_data)-time_step-intervals):
        x = normalize_data[i:i+time_step]
        y = normalize_data[i+time_step+intervals]
        data_x.append(x)
        data_y.append(y)

    # split into train and test sets
    train_x = np.array(data_x[0:len(data_x)-288])                   
    train_y = np.array(data_y[0:len(data_x)-288])
    test_x = np.array(data_x[len(data_x)-288:])
    test_y = np.array(data_y[len(data_y)-288:])
    return train_x,train_y,test_x,test_y

    # # split into train and test sets
    # train_x = np.array(data_x[0:len(data_x)-576])                   
    # train_y = np.array(data_y[0:len(data_x)-576])
    # test_x = np.array(data_x[len(data_x)-576:])
    # test_y = np.array(data_y[len(data_y)-576:])
    # return train_x,train_y,test_x,test_y


def lstm_model_twolayer(input_dim = 1,out_dim = 1,timesteps = 4,unit1 = 32, unit2 = 100):
                               # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(unit1, return_sequences = True, input_shape = (timesteps, input_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(unit2))
    model.add(Dense(out_dim, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer ='rmsprop')

    return model

def MAPE(y_pre,y_test): 
    mape = np.sum(np.fabs((y_test-y_pre)/y_test))/len(y_test)
    return mape

def Evaluate_TestData(predict_y,test_y):
    mse_score = mean_squared_error(predict_y,test_y)
    mae_score = mean_absolute_error(predict_y,test_y)
    mape_score = MAPE(predict_y,test_y)
    RMSE = np.sqrt(mse_score)
    r = "mean RMSE: %.4f" % RMSE
    m = "mean MAE: %.4f" % mae_score
    ma = "mean MAPE: %.4f" % mape_score
    return r,m,ma
    print('result：')
    print('---------------------')
    print("mean RMSE: %.4f" % RMSE)
    print("mean MAE: %.4f" % mae_score)
    print("mean MAPE: %.4f" % mape_score)
    print('---------------------')
    
f = open('C:/Users/panyy/Desktop/LSTM预测/CA_I405_bottleneck_13.51_train.csv')
file = pd.read_csv(f)
ndata = file['Flow per lane']           
data = ndata[:,np.newaxis]
# normalize the dataset
scale = MinMaxScaler(feature_range = (0,1))
normalize_data=scale.fit_transform(data) 
    
model = lstm_model_twolayer(input_dim = 1,out_dim = 1)
train_x,train_y,test_x,test_y=train_test_split_1dim(normalize_data,time_step = 4,intervals = 1)
model.fit(train_x, train_y, batch_size = 1, epochs = 100)
predictions=model.predict(test_x)
p=scale.inverse_transform(predictions)
y=scale.inverse_transform(test_y)
result=Evaluate_TestData(p,y)

# To store the test and the predicted results
pd.DataFrame(p).to_csv('Prediction_Results_twolayers.csv')

#plot
plt.plot(p,'r',label='prediction')
plt.plot(y,'g',label='test')
plt.xlabel(result)
plt.ylabel('flow (veh)')
plt.legend(loc='upper left')
plt.title('LSTM_twolayer')
plt.savefig('LSTM_twolayers.jpg')
plt.show()