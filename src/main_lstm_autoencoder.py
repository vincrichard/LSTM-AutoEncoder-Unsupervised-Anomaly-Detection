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
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
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
    print('====> Validation Average loss: {:.4f}'.format(avg_loss))
    # Checkpoint
    if validation:
        loss_checkpoint_valid.losses.append(avg_loss)
        model_management.checkpoint(epoch, model, optimizer, avg_loss)
        return earlyStopping.check_training(avg_loss)


def compare_ts(model, loader):
    """
    Plot a random original multivariate sequence from loader and the predict one
    """
    ind = np.random.randint(0, loader.batch_size)
    model.eval()
    with torch.no_grad():
        for id_batch, data in enumerate(loader):
            data = data.to(device)
            output = model.forward(data)
            loss = criterion(data[:, ind, :], output[:, ind, :].to(device))
            plt.plot(data[:, ind, :].cpu().numpy())
            plt.plot(output[:, ind, :].numpy())
            plt.title('loss= %f' % loss.item())
            plt.legend(['data mean', 'data max', 'data min',
                        'predict mean', 'predict max', "predict min"],
                       bbox_to_anchor=(1.1, 1.05))
            break


if __name__ == "__main__":
    for epoch in range(1, 5):
        train(epoch)
        if evaluate(valid_loader, validation=True, epoch=epoch):
            break
