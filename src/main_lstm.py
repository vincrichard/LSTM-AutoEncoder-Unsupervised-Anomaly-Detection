import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.airbus_data import AirbusDataSeq, DataSeq
from src.model.LSTM import LSTMVanilla
from src.utils.callbacks import EarlyStopping
from src.utils.model_management import ModelManagement
from src.utils.plot_class import LossCheckpoint

name_model='lstm_model'
path_model='../models/'
nb_feature=1
seq_length=20
batch_size=32
device = torch.device('cpu')
##############
#load dataset#
##############

# train_dataset = AirbusDataSeq('../data/seq/seq_', range(0))
test_dataset = AirbusDataSeq('../data/seq/seq_', range(1))
#Create Data generator
# train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)


##############
# build models#
##############
model = LSTMVanilla(input_size=1, hidden_size=100, num_layers=2, pred_length=1, batch_size=batch_size, device=device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
#loss
criterion = torch.nn.MSELoss()
# def loss_function(x, x_rec):
#     return F.mse_loss(x, x_rec)
#Callbacks
earlyStopping = EarlyStopping(patience=5)
model_management = ModelManagement(path_model, name_model)
#Plot
plot_loss = LossCheckpoint()


def train(epoch, loader):
    model.train()
    train_loss = 0
    for seq_number, inout_seq in enumerate(loader):
        batch_loader = DataLoader(dataset=DataSeq(inout_seq.reshape(-1, seq_length + 1, 1)), batch_size=batch_size,
                                  shuffle=False, drop_last=True)  # .reshape(-1, seq_length+1, nb_feature)
        model.reset_hidden()
        for batch_seq, inout_seq_batch in enumerate(batch_loader):
            x = inout_seq_batch[:, :-1, :].to(device)
            y = inout_seq_batch[:, -1, :].to(device)

            optimizer.zero_grad()

            pred = model(x.float())
            loss = criterion(y, pred)
            # loss = loss_function(y.to(device), pred)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            # reset hidden and cell state
            model.detach_hidden()
            print('\r', 'Seq [{}/{} ({:.0f}%)] Ongoing [{}/{} ({:.0f}%\tLoss: {:.6f})]'.format(
                seq_number + 1, len(loader),
                (seq_number + 1) * 100 / len(loader),
                batch_seq + 1, len(batch_loader),
                (batch_seq + 1) * 100 / len(batch_loader), loss.item()
            ), sep='', end='', flush=True)

        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, seq_number + 1, len(loader),
        #     (seq_number + 1) / len(loader) * 100, loss.item() / len(data)))
    avg_loss = train_loss / (len(batch_loader) * batch_size)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    model_management.checkpoint(epoch, model, optimizer, avg_loss)
    if epoch % 2 == 1:
        model_management.save_best_model()
    plot_loss.losses.append(avg_loss)
    return earlyStopping.check_training(avg_loss)


if __name__ == "__main__":
    for epoch in range(1, 5):
        if train(epoch, test_loader):
            break
    model_management.save_best_model()
    plot_loss.plot()
    # for seq_number, inout_seq in enumerate(test_loader):
    #     torch.save(inout_seq, '../data/seq/seq_%d.pt'%(seq_number))
