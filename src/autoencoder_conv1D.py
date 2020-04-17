import torch.nn as nn


class AutoEncoderConv1D(nn.Module):
    def __init__(self, x_dim, conv_dim1, kernel_lenght, h_dim1, h_dim2, z_dim):
        super(AutoEncoderConv1D, self).__init__()

        # encoder part
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_dim1, kernel_size=(1, kernel_lenght), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(conv_dim1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features=(x_dim-kernel_lenght +1)*conv_dim1, out_features=h_dim1, bias=True),
            nn.BatchNorm1d(h_dim1),
            nn.ReLU(True),
            nn.Linear(in_features=h_dim1, out_features=h_dim2, bias=True),
            nn.BatchNorm1d(h_dim2),
            nn.ReLU(True),
            nn.Linear(in_features=h_dim2, out_features=z_dim, bias=True)
        )

        # decoder part
        self.decoder = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=h_dim2, bias=True),
            nn.BatchNorm1d(h_dim2),
            nn.ReLU(True),
            nn.Linear(in_features=h_dim2, out_features=h_dim1, bias=True),
            nn.BatchNorm1d(h_dim1),
            nn.ReLU(True),
            nn.Linear(in_features=h_dim1, out_features=x_dim, bias=True)
        )

    def forward(self, x):
        z = self.encoder(x)
        recover = self.decoder(z)
        return recover