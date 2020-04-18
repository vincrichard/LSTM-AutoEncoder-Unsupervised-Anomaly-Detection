import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.autoencoder_vanilla import AutoEncoderVanilla
from src.callbacks import EarlyStopping
from src.airbus_data import AirbusData
from src.model_management import ModelManagement
from src.plot_class import PlotLoss
from src.autoencoder_conv1D import AutoEncoderConv1D

batch_size=8
z_dim=30
name_file='encoded_data'
name_model='encoder_model'
path_model='../model/'

##############
#load dataset#
##############
def scale(x):
  return (x-np.mean(x))/np.std(x)

# train_dataset = pd.read_csv('../data/airbus_train.csv')
test_dataset = AirbusData('../data/airbus_test.csv',nrows=16) #transform=scale
#Create Data generator
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

##############
# build model#
##############
model = AutoEncoderVanilla(x_dim=61440, h_dim1= 512, h_dim2=256, z_dim=z_dim)
# model = AutoEncoderConv1D(x_dim=61440, conv_dim1=64, kernel_length=300, stride=30, h_dim1=512, h_dim2=256, z_dim=z_dim)
# if torch.cuda.is_available():
#     autoencoder_vanilla.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

#Callbacks
earlyStopping = EarlyStopping(patience=5)
model_management = ModelManagement(path_model, name_model)
#Plot
plot_loss = PlotLoss()


def loss_function(x, x_rec):
    return F.mse_loss(x, x_rec)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(test_loader):
        # data = data.cuda()
        optimizer.zero_grad()

        # x_rec = model.forward(data.float())
        x_rec = model.forward(data.view(-1, 1, 1, 61440).float())
        loss = loss_function(data, x_rec)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx+1, len(test_loader),
            (batch_idx+1) / len(test_loader)*100, loss.item() / len(data)))
    avg_loss = train_loss / len(test_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    model_management.checkpoint(epoch, model, optimizer, avg_loss)
    plot_loss.losses.append(avg_loss)
    return earlyStopping.check_training(avg_loss)

def get_feature_reduction(save_file):
    with torch.no_grad():
        data_reduction = None
        for batch_idx, data in enumerate(test_loader):
            z = model.encoder(data.float())
            if not torch.is_tensor(data_reduction):
                data_reduction = z
            else:
                data_reduction = torch.cat([data_reduction, z], dim=0)
            if save_file == True:
                data_to_save = data_reduction.numpy()
                pd.DataFrame(data_to_save).to_csv("../data/encoded/%s.csv"%name_file)

if __name__ == "__main__":
    for epoch in range(1, 5):
        if train(epoch):
            break
    get_feature_reduction(True)
    model_management.save(model)
    plot_loss.plot()

    #Load model
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()