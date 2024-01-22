from LinearModel import LinearFootballPredictionModel
from RNNModel import RNNFootballPredictionModel
from LSTMModel import LSTMFootballPredictionModel
from import_data_to_objects import getObjects
import datetime
import torch
import pandas as pd

def calc_acc(pred, act):
    option =  0
    acc = 0
    for i in range(len(act)):
        if act[i] == 1:
            option = i
            break
    prediction = pred.index(max(pred))
    if prediction == option: 
        if pred[option] > 0.75:
            return 1
        elif pred[option] > 0.5 and pred[option] < 0.75:
            return 0.5 + (pred[option] - 0.5) * 0.02
        else:
            return 0.5
    else:
        return 0
    
def calc_acc2(pred, act):
    option =  0
    acc = 0
    for i in range(len(act)):
        if act[i] == 1:
            option = i
            break
    prediction = pred.index(max(pred))
    if prediction == option: 
        return 1
    else:
        return 0
    

linearModel = LinearFootballPredictionModel()
RNNModel = RNNFootballPredictionModel()
LSTMModel = LSTMFootballPredictionModel()

linearModel.load_state_dict(torch.load('models/v1/Linear_model.pth'), strict=False)
RNNModel.load_state_dict(torch.load('models/v1/RNN_model.pth'), strict=False)
LSTMModel.load_state_dict(torch.load('models/v1/LSTM_model.pth'), strict=False)
linearModel.to('cuda')
RNNModel.to('cuda')
LSTMModel.to('cuda')
linearModel.eval()
RNNModel.eval()
LSTMModel.eval()
models = {'Linear': linearModel, 'RNN': RNNModel, 'LSTM': LSTMModel}
acc = {'Linear': 0, 'RNN': 0, 'LSTM': 0}
acc2 = {'Linear': 0, 'RNN': 0, 'LSTM': 0}

obj = getObjects(date=datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'), to=0)
with torch.no_grad():

#     # Make predictions
    for object in obj:
        for m in models:
            object_t = object.to_single_tensor()
            data = torch.cat([object_t[:,0,0], object_t[0,1:,0], object_t[1,1:,0]])
            predictions = models[m](prediction_data = data)
            acc[m] = acc[m] + calc_acc(predictions.tolist(), object.getOutput().tolist())
            acc2[m] = acc2[m] + calc_acc2(predictions.tolist(), object.getOutput().tolist())
    for m in models:
        acc[m] = acc[m]/len(obj)
        acc2[m] = acc2[m]/len(obj)
    print(acc)
    print(acc2)

