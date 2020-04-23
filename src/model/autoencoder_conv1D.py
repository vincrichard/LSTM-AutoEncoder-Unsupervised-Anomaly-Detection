import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        x = x.view(*self.shape)
        return x


class AutoEncoderConv1D(nn.Module):
    def __init__(self, x_dim, conv_dim1, kernel_length, stride, h_dim1, h_dim2, z_dim):
        super(AutoEncoderConv1D, self).__init__()

        # encoder part
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_dim1, kernel_size=(1, kernel_length), stride=stride, padding=0,
                      bias=True),
            nn.BatchNorm2d(conv_dim1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features=(int((x_dim - kernel_length) / stride) + 1) * conv_dim1, out_features=h_dim1,
                      bias=True),
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
            nn.Linear(in_features=h_dim1, out_features=(int((x_dim - kernel_length) / stride) + 1) * conv_dim1,
                      bias=True),
            nn.View((conv_dim1, int((x_dim - kernel_length) / stride) + 1)),
            nn.ConvTranspose1d(in_channels=conv_dim1, out_channels=1, kernel_size=(1, kernel_length), stride=stride)
        )

    def forward(self, x):
        z = self.encoder(x)
        recover = self.decoder(z)
        return recover
