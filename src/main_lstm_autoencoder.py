import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from src.utils.airbus_data import AirbusData
from src.model.LSTM_auto_encoder import LSTMAutoEncoder
from src.utils.callbacks import EarlyStopping
from src.utils.model_management import ModelManagement
from src.utils.plot_class import LossCheckpoint
from sklearn.metrics.pairwise import cosine_similarity

step_window = 30
size_window = 50
name_model = 'lstm_model'
path_model = '../models/'
batch_size = 32
device = torch.device('cpu')

# ----------------------------------#
#         load dataset              #
# ----------------------------------#

train_seq = AirbusData('../data/train_seq_mean_max_min.pt', type='pytorch')
train_loader = DataLoader(train_seq, batch_size, shuffle=True)
seq_length, size_data, nb_feature = train_seq.data.shape

valid_seq = AirbusData('../data/valid_seq_mean_max_min.pt', type='pytorch')
valid_loader = DataLoader(valid_seq, batch_size, shuffle=True)

test_seq = AirbusData('../data/test_seq_mean_max_min.pt', type='pytorch')
test_loader = DataLoader(test_seq, batch_size, shuffle=False)

# ----------------------------------#
#         build model              #
# ----------------------------------#

model = LSTMAutoEncoder(num_layers=1, hidden_size=100, nb_feature=nb_feature, device=device)
model = model.to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss
criterion = torch.nn.MSELoss()
# Callbacks
earlyStopping = EarlyStopping(patience=5)
model_management = ModelManagement(path_model, name_model)
# Plot
loss_checkpoint_train = LossCheckpoint()
loss_checkpoint_valid = LossCheckpoint()


def train(epoch):
    model.train()
    train_loss = 0
    for id_batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        # forward
        data = data.to(device)
        output = model.forward(data)
        loss = criterion(data, output.to(device))
        # backward
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()

        print('\r', 'Training [{}/{} ({:.0f}%)] \tLoss: {:.6f})]'.format(
            id_batch + 1, len(train_loader),
            (id_batch + 1) * 100 / len(train_loader),
            loss.item()), sep='', end='', flush=True)

    avg_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, avg_loss))
    loss_checkpoint_train.losses.append(avg_loss)


def evaluate(loader, validation=False, epoch=0):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for id_batch, data in enumerate(loader):
            data = data.to(device)
            output = model.forward(data)
            loss = criterion(data, output.to(device))
            eval_loss += loss.item()
        print('\r', 'Eval [{}/{} ({:.0f}%)] \tLoss: {:.6f})]'.format(
            id_batch + 1, len(loader),
            (id_batch + 1) * 100 / len(loader),
            loss.item()), sep='', end='', flush=True)
    avg_loss = eval_loss / len(loader)
    print('====> Validation Average loss: {:.6f}'.format(avg_loss))
    # Checkpoint
    if validation:
        loss_checkpoint_valid.losses.append(avg_loss)
        model_management.checkpoint(epoch, model, optimizer, avg_loss)
        return earlyStopping.check_training(avg_loss)


def compare_seq(model, seq):
    """
    Plot a the original multivariate sequence from loader and the predict one
    """
    model.eval()
    with torch.no_grad():
        seq = seq.to(device)
        output = model.forward(seq.reshape(1, seq_length, -1).to(device))
        loss = criterion(seq.reshape(1, seq_length, -1)[0, :, :], output[0, :, :].to(device))
        plt.plot(seq.cpu().numpy())
        plt.plot(output[0, :, :].numpy())
        plt.title('loss= %f' % loss.item())
        plt.legend(['data mean', 'data max', 'data min', 'trend', 'kurtosis', 'var', 'level_shift',
                      'predict mean', 'predict max', "predict min", 'predict trend', 'predict kurtosis', 'predict var', 'predict level_shift'],
                      bbox_to_anchor=(1.1, 1.05))

def predict(loader, input_data, model):
    eval_loss=0
    model.eval()
    predict = torch.zeros(size=input_data.shape, dtype=torch.float)
    with torch.no_grad():
        for id_batch, data in enumerate(loader):
            data = data.to(device)
            output = model.forward(data)
            predict[id_batch*data.shape[0]:(id_batch+1)*data.shape[0], :, :] = output.reshape(data.shape[0],seq_length, -1)
            loss = criterion(data, output.to(device))
            eval_loss += loss.item()

    avg_loss = eval_loss / len(loader)
    print('====> Prediction Average loss: {:.6f}'.format(avg_loss))
    return predict

def get_avg_pred(loader, input_data, model, epoch=10):
    pred =  predict(loader, input_data, model)
    for i in range(epoch):
        pred = pred + predict(loader, input_data, model)
    return pred/(epoch+1)

def get_original_ts(data, data_type):
    """
    Recreate the original time serie leaving the overlapping part of the seq since the last part are the begnning of the decoding
    keeping the first element give more potential error on prediction
    """
    last_values = size_window - step_window
    if data_type=='train':
        nb_ts_origine=1677
    elif data_type=='test':
        nb_ts_origine=2511
    else:
        raise ValueError('data_type value is wrong: ', data_type)
    return torch.cat((data[:, :step_window, :].reshape(nb_ts_origine, -1, nb_feature),
                          data.reshape(nb_ts_origine, -1, nb_feature)[:,-last_values:,:]),axis=1)

def calcul_score_serie(data, predict, data_type, calc_type='max'):
    """
    Calculate the error of prediction with different possible technique
    """
    if data_type=='train':
        nb_ts_origine=1677
    elif data_type=='test':
        nb_ts_origine=2511
    else:
        raise ValueError('data_type value is wrong: ', data_type)
    if calc_type=='mean':
        loss =  torch.pow(torch.add(get_original_ts(data, data_type=data_type),
                                  get_original_ts(-predict, data_type=data_type)),
                         2).numpy()
        loss = np.mean(loss, axis=1)
        return loss
    elif calc_type=='abs_mean':
        loss = (torch.div(get_original_ts(-predict, data_type=data_type),
                        get_original_ts(data, data_type=data_type)+1*10**-8) -1).numpy()
        loss = np.mean(loss, axis=1)
        return loss
    elif calc_type=='cos_similarity':
        loss = np.zeros((nb_ts_origine, data.shape[2]))
        for i in range(data.shape[2]):
            loss[:, i] = 1 - np.mean(cosine_similarity(get_original_ts(data, data_type=data_type).numpy()[:, :, i], get_original_ts(predict, data_type=data_type).numpy()[:, :, i]), axis = 1)
        return loss
    else:
        raise ValueError('calc_type value is wrong: ', calc_type)

if __name__ == "__main__":
    for epoch in range(1, 5):
        train(epoch)
        if evaluate(valid_loader, validation=True, epoch=epoch):
            break
        # Lr on plateau
        if earlyStopping.patience_count == 5:
            print('lr on plateau ', optimizer.param_groups[0]['lr'], ' -> ', optimizer.param_groups[0]['lr'] / 10)
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10