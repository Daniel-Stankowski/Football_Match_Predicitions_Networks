import torch
import torch.nn as nn
import torch.optim as optim
from RNNModel import RNNFootballPredictionModel
from LSTMModel import LSTMFootballPredictionModel
from LinearModel import LinearFootballPredictionModel
from torch.utils.data import DataLoader, TensorDataset
from import_data_to_objects import getObjects
import datetime

# Assuming you have training_data and target_data as your training dataset
# You need to convert them to PyTorch tensors or datasets
train_objects = getObjects(date=datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'), to=1)
val_objects = getObjects(date=datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'), to=0)
train_dataset = list(map(lambda x: x.to_single_tensor(), train_objects))
val_dataset = list(map(lambda x: x.to_single_tensor(), val_objects))
train_outputs = list(map(lambda x: x.getOutput(), train_objects))
val_outputs = list(map(lambda x: x.getOutput(), val_objects))

train_list = list(zip(train_dataset, train_outputs))
val_list = list(zip(val_dataset, val_outputs))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

train_tensordataset = TensorDataset(*map(torch.stack, zip(*train_list)))
val_tensordataset = TensorDataset(*map(torch.stack, zip(*val_list)))

batch_size = 1
train_loader = DataLoader(train_tensordataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_tensordataset, batch_size=batch_size, shuffle=True)

models = {'RNN': RNNFootballPredictionModel().to(device), 'LSTM': LSTMFootballPredictionModel().to(device)}
criterion = nn.CrossEntropyLoss()
num_epochs = 10
for m in models:
    model = models[m]
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    epochs = ''
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for i, (batch_inputs, batch_labels) in enumerate(train_loader):
            if i % 1000 == 0:
                num = int(i/1000)
                print(f'Epoch {epoch + 1}/{num_epochs}, training {num*1000}-{(num+1)*1000 - 1}')
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(input_data = batch_inputs[0])
            outputs = outputs.cuda()
            # Compute the loss
            # print(outputs)
            loss = criterion(outputs, batch_labels[0].cuda())
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (batch_inputs, batch_labels) in enumerate(train_loader):
                outputs = model(input_data = batch_inputs[0])
                loss = criterion(outputs, batch_labels[0].cuda())
                val_loss += loss.item()
    # Print training statistics
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        epochs = epochs + f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}\n'
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for i, (batch_inputs, batch_labels) in enumerate(train_loader):
            if i % 1000 == 0:
                num = int(i/1000)
                print(f'Epoch {epoch + 1}/{num_epochs}, training embedding {num*1000}-{(num+1)*1000 - 1}')
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            batch_inputs = batch_inputs[0]
            data = torch.cat([batch_inputs[:,0,0], batch_inputs[0,1:,0], batch_inputs[1,1:,0]])
            outputs = model(prediction_data = data)
            outputs = outputs.cuda()
            # Compute the loss
            # print(outputs)
            loss = criterion(outputs, batch_labels[0].cuda())
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for i, (batch_inputs, batch_labels) in enumerate(val_loader):
                batch_inputs = batch_inputs[0]
                data = torch.cat([batch_inputs[:,0,0], batch_inputs[0,1:,0], batch_inputs[1,1:,0]])
                outputs = model(prediction_data = data)
                loss = criterion(outputs, batch_labels[0].cuda())
                val_loss += loss.item()
        # Print training statistics
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
        epochs = epochs + f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}\n'

    torch.save(model.state_dict(), f'{m}_model.pth')
    text_file = open(f'{m}_log.txt', "w")
    text_file.write(epochs)
    text_file.close()