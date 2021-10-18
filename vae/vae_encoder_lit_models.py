import torch
from torch import nn
import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)

def load_model(config, network, number_classes, data, seq_length, y_encoder_only):
    if network == "deep":
        selected_network = Deep(config, number_classes, data, seq_length, y_encoder_only)
    elif network == "conv":
        selected_network = Conv(config, number_classes, data, seq_length, y_encoder_only)
    elif network == "LSTM":
        selected_network = LSTM(config, number_classes, data, seq_length, y_encoder_only)
    elif network == "Resnet":
        selected_network = Resnet(config, number_classes, data, seq_length, y_encoder_only)
    elif network == "GPT2":
        selected_network = GPT2(config, number_classes, data, seq_length, y_encoder_only)
    elif network == "Resnet1d":
        selected_network = Resnet1d(config, number_classes, data, seq_length, y_encoder_only)
    else:
        print("no network")
        selected_network = []

    return selected_network


class Deep(pl.LightningModule): # original nn.module
    def __init__(self, config, number_classes, data, seq_length, y_encoder):
        global final_layer_size
        super(Deep, self).__init__()

        # clean if https://medium.com/swlh/3-alternatives-to-if-statements-to-make-your-python-code-more-readable-91a9991fb353
        self.number_classes = number_classes
        self.number_of_layers = config["enc_number_of_layers"]
        self.layer_1_size = config["enc_layer_1_size"]
        self.layer_2_size = config["enc_layer_2_size"]
        self.layer_3_size = config["enc_layer_3_size"]
        self.layer_4_size = config["enc_layer_4_size"]
        self.layer_5_size = config["enc_layer_5_size"]
        self.layer_6_size = config["enc_layer_6_size"]
        self.layer_7_size = config["enc_layer_7_size"]
        self.relu_on = config["enc_activation"]
        self.layer_1_bias = config["enc_layer_1_bias"]
        self.layer_2_bias = config["enc_layer_2_bias"]
        self.layer_3_bias = config["enc_layer_3_bias"]
        self.layer_4_bias = config["enc_layer_4_bias"]
        self.layer_5_bias = config["enc_layer_5_bias"]
        self.layer_6_bias = config["enc_layer_6_bias"]
        self.drop_0_n = config["enc_drop_0"]
        self.drop_1_n = config["enc_drop_1"]
        self.drop_2_n = config["enc_drop_2"]
        self.drop_3_n = config["enc_drop_3"]
        self.drop_4_n = config["enc_drop_4"]
        self.latent_space = config["latent_space"]
        first_layer_size = seq_length*4

        self.layer_0_y= nn.Linear(15, 4)
        self.batch_norm_0_y = nn.BatchNorm1d(4)
        self.y_encoder=y_encoder

        if self.number_of_layers == 2:
            self.layer_0 = nn.Linear(first_layer_size, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            if self.y_encoder == True:
                self.layer_1_mu = nn.Linear(self.layer_1_size+4, self.latent_space, bias=self.layer_1_bias)
                self.layer_1_var = nn.Linear(self.layer_1_size+4, self.latent_space, bias=self.layer_1_bias)
            else:
                self.layer_1_mu = nn.Linear(self.layer_1_size, self.latent_space, bias=self.layer_1_bias)
                self.layer_1_var = nn.Linear(self.layer_1_size, self.latent_space, bias=self.layer_1_bias)

        elif self.number_of_layers == 3:
            self.layer_0 = nn.Linear(first_layer_size, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
            self.batch_norm_1 = nn.BatchNorm1d(self.layer_2_size)
            if self.y_encoder == True:
                self.layer_2_mu = nn.Linear(self.layer_2_size+4, self.latent_space)
                self.layer_2_var = nn.Linear(self.layer_2_size+4, self.latent_space)
            else:
                self.layer_2_mu = nn.Linear(self.layer_2_size, self.latent_space)
                self.layer_2_var = nn.Linear(self.layer_2_size, self.latent_space)

        elif self.number_of_layers == 4:
            self.layer_0 = nn.Linear(first_layer_size, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
            self.batch_norm_1 = nn.BatchNorm1d(self.layer_2_size)
            self.layer_2 = nn.Linear(self.layer_2_size, self.layer_3_size, bias=self.layer_2_bias)
            self.batch_norm_2 = nn.BatchNorm1d(self.layer_3_size)
            self.layer_3_mu = nn.Linear(self.layer_3_size, self.latent_space)
            self.layer_3_var = nn.Linear(self.layer_3_size, self.latent_space)

        elif self.number_of_layers == 5:
            self.layer_0 = nn.Linear(first_layer_size, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
            self.batch_norm_1 = nn.BatchNorm1d(self.layer_2_size)
            self.layer_2 = nn.Linear(self.layer_2_size, self.layer_3_size, bias=self.layer_2_bias)
            self.batch_norm_2 = nn.BatchNorm1d(self.layer_3_size)
            self.layer_3 = nn.Linear(self.layer_3_size, self.layer_4_size, bias=self.layer_3_bias)
            self.batch_norm_3 = nn.BatchNorm1d(self.layer_4_size)
            self.layer_4_mu = nn.Linear(self.layer_4_size, self.latent_space)
            self.layer_4_var = nn.Linear(self.layer_4_size, self.latent_space)

        elif self.number_of_layers == 6:
            self.layer_0 = nn.Linear(first_layer_size, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
            self.batch_norm_1 = nn.BatchNorm1d(self.layer_2_size)
            self.layer_2 = nn.Linear(self.layer_2_size, self.layer_3_size, bias=self.layer_2_bias)
            self.batch_norm_2 = nn.BatchNorm1d(self.layer_3_size)
            self.layer_3 = nn.Linear(self.layer_3_size, self.layer_4_size, bias=self.layer_3_bias)
            self.batch_norm_3 = nn.BatchNorm1d(self.layer_4_size)
            self.layer_4 = nn.Linear(self.layer_4_size, self.layer_5_size, bias=self.layer_4_bias)
            self.batch_norm_4 = nn.BatchNorm1d(self.layer_5_size)
            self.layer_5_mu = nn.Linear(self.layer_5_size, self.latent_space)
            self.layer_5_var = nn.Linear(self.layer_5_size, self.latent_space)

        self.activation = torch.nn.ReLU() if self.relu_on == "relu" else torch.nn.Tanh()
        self.drop_0 = nn.Dropout(self.drop_0_n)
        self.drop_1 = nn.Dropout(self.drop_1_n)
        self.drop_2 = nn.Dropout(self.drop_2_n)
        self.drop_3 = nn.Dropout(self.drop_3_n)
        self.drop_4 = nn.Dropout(self.drop_4_n)
        self.softmax = nn.LogSoftmax()


    def forward(self, x , y):
        y = self.layer_0_y(y)
        y = self.batch_norm_0_y(y)
        y = self.activation(y)

        if self.number_of_layers == 2:
            x = self.layer_0(x)
            x = self.batch_norm_0(x)
            x = self.activation(x)
            x = self.drop_0(x)
            if self.y_encoder == True:
                x = torch.cat([x, y])
            x_mu = self.layer_1_mu(x)
            x_var = self.layer_1_var(x)
            #x = self.softmax(x)

        elif self.number_of_layers == 3:
            x = self.layer_0(x)
            x = self.batch_norm_0(x)
            x = self.activation(x)
            x = self.drop_0(x)
            x = self.layer_1(x)
            x = self.batch_norm_1(x)
            x = self.activation(x)
            x = self.drop_1(x)
            if self.y_encoder == True:
                x = torch.cat([x, y], dim=1)
            x_mu = self.layer_2_mu(x)
            x_var = self.layer_2_var(x)
            #x = self.softmax(x)

        elif self.number_of_layers == 4:
            x = self.layer_0(x)
            x = self.batch_norm_0(x)
            x = self.activation(x)
            x = self.drop_0(x)
            x = self.layer_1(x)
            x = self.batch_norm_1(x)
            x = self.activation(x)
            x = self.drop_1(x)
            x = self.layer_2(x)
            x = self.batch_norm_2(x)
            x = self.activation(x)
            x = self.drop_2(x)
            x_mu = self.layer_3_mu(x)
            x_var = self.layer_3_var(x)
            #x = self.softmax(x)

        elif self.number_of_layers == 5:
            x = self.layer_0(x)
            x = self.batch_norm_0(x)
            x = self.activation(x)
            x = self.drop_0(x)

            x = self.layer_1(x)
            x = self.batch_norm_1(x)
            x = self.activation(x)
            x = self.drop_1(x)

            x = self.layer_2(x)
            x = self.batch_norm_2(x)
            x = self.activation(x)
            x = self.drop_2(x)

            x = self.layer_3(x)
            x = self.batch_norm_3(x)
            x = self.activation(x)
            x = self.drop_3(x)

            x_mu = self.layer_4_mu(x)
            x_var = self.layer_4_var(x)
            #x = self.softmax(x)


        elif self.number_of_layers == 6:
            x = self.layer_0(x)
            x = self.batch_norm_0(x)
            x = self.activation(x)
            x = self.drop_0(x)

            x = self.layer_1(x)
            x = self.batch_norm_1(x)
            x = self.activation(x)
            x = self.drop_1(x)

            x = self.layer_2(x)
            x = self.batch_norm_2(x)
            x = self.activation(x)
            x = self.drop_2(x)

            x = self.layer_3(x)
            x = self.batch_norm_3(x)
            x = self.activation(x)
            x = self.drop_3(x)

            x = self.layer_4(x)
            x = self.batch_norm_4(x)
            x = self.activation(x)
            x = self.drop_4(x)

            x_mu = self.layer_5_mu(x)
            x_var = self.layer_5_var(x)
            #x = self.softmax(x)

        return x_mu, x_var


class Conv(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_encoder):
        super(Conv, self).__init__()

        self.number_of_layers = config["number_of_layers"]
        self.channel_1_size = config["channel_1_size"]
        self.channel_2_size = config["channel_2_size"]
        self.channel_3_size = config["channel_3_size"]
        self.channel_4_size = config["channel_4_size"]
        self.channel_5_size = config["channel_5_size"]
        self.kernel_1_size = config["kernel_1_size"]
        self.kernel_2_size = config["kernel_2_size"]
        self.kernel_3_size = config["kernel_3_size"]
        self.kernel_4_size = config["kernel_4_size"]
        self.kernel_5_size = config["kernel_5_size"]
        self.stride_1_size = config["stride_1_size"]
        self.stride_2_size = config["stride_2_size"]
        self.stride_3_size = config["stride_3_size"]
        self.stride_4_size = config["stride_4_size"]
        self.stride_5_size = config["stride_5_size"]

        if self.number_of_layers == 2:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=4,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=False)

        elif self.number_of_layers == 3:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=False)
            self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=4,
                                              kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0,
                                              bias=False)

        elif self.number_of_layers == 4:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=False)
            self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=self.channel_3_size,
                                              kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0,
                                              bias=False)
            self.deconv3 = nn.ConvTranspose1d(in_channels=self.channel_3_size, out_channels=4,
                                              kernel_size=self.kernel_3_size, stride=self.stride_3_size, padding=0,
                                              bias=False)

        elif self.number_of_layers == 5:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=False)
            self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=self.channel_3_size,
                                              kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0,
                                              bias=False)
            self.deconv3 = nn.ConvTranspose1d(in_channels=self.channel_3_size, out_channels=self.channel_4_size,
                                              kernel_size=self.kernel_3_size, stride=self.stride_3_size, padding=0,
                                              bias=False)
            self.deconv4 = nn.ConvTranspose1d(in_channels=self.channel_4_size, out_channels=4,
                                              kernel_size=self.kernel_4_size, stride=self.stride_4_size, padding=0,
                                              bias=False)

        elif self.number_of_layers == 6:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=False)
            self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=self.channel_3_size,
                                              kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0,
                                              bias=False)
            self.deconv3 = nn.ConvTranspose1d(in_channels=self.channel_3_size, out_channels=self.channel_4_size,
                                              kernel_size=self.kernel_3_size, stride=self.stride_3_size, padding=0,
                                              bias=False)
            self.deconv4 = nn.ConvTranspose1d(in_channels=self.channel_4_size, out_channels=self.channel_5_size,
                                              kernel_size=self.kernel_4_size, stride=self.stride_4_size, padding=0,
                                              bias=False)
            self.deconv5 = nn.ConvTranspose1d(in_channels=self.channel_5_size, out_channels=4,
                                              kernel_size=self.kernel_5_size, stride=self.stride_5_size, padding=0,
                                              bias=False)

        '''
        self.linear1 = nn.Linear(3, 4)
        self.deconv1 = nn.ConvTranspose1d(in_channels=4, out_channels=self.channel_1_size, kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0, bias=False)
        self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size, kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0, bias=False)
        self.deconv3 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=self.channel_3_size, kernel_size=self.kernel_3_size, stride=self.stride_3_size, padding=0, bias=False)
        self.deconv4 = nn.ConvTranspose1d(in_channels=self.channel_3_size, out_channels=self.channel_4_size, kernel_size=self.kernel_4_size, stride=self.stride_4_size, padding=0, bias=False)
        self.deconv5 = nn.ConvTranspose1d(in_channels=self.channel_4_size, out_channels=4, kernel_size=self.kernel_5_size, stride=self.stride_5_size, padding=0, bias=False)
        '''

    def forward(self, x , y):

        if self.number_of_layers == 2:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = torch.nn.functional.interpolate(x, size=500)

        elif self.number_of_layers == 3:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = torch.nn.functional.interpolate(x, size=500)

        elif self.number_of_layers == 4:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = torch.nn.functional.interpolate(x, size=500)

        elif self.number_of_layers == 5:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
            x = torch.nn.functional.interpolate(x, size=500)

        elif self.number_of_layers == 6:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
            x = self.deconv5(x)
            x = torch.nn.functional.interpolate(x, size=500)

            '''
            x = self.linear1(x)
            x = x.view(-1, 4, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
            x = self.deconv5(x)
            x = torch.nn.functional.interpolate(x, size = 500)
            '''

        return x


class LSTM(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_encoder):
        super(LSTM, self).__init__()

        self.number_classes = number_classes
        self.number_of_layers = config["number_of_layers"]

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.layer_3_size = config["layer_3_size"]
        self.layer_4_size = config["layer_4_size"]

        self.LSTM_layers_1 = config["LSTM_layers_1"]
        self.LSTM_layers_2 = config["LSTM_layers_2"]
        self.state_instead_input = config["state_instead_input"]

        if self.state_instead_input == True:

            if self.number_of_layers == 2:
                self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
                self.rnn1 = nn.LSTM(self.layer_1_size, self.layer_1_size, 1, batch_first=True)
                self.linear2 = nn.Linear(self.layer_1_size, 2000)

            else:
                self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
                self.rnn1 = nn.LSTM(self.layer_1_size, self.layer_1_size, 1, batch_first=True)
                self.linear2 = nn.Linear(self.layer_1_size, self.layer_2_size)
                self.rnn2 = nn.LSTM(self.layer_2_size, self.layer_2_size, 1, batch_first=True)
                self.linear3 = nn.Linear(self.layer_2_size, 2000)

        else:
            if self.number_of_layers == 2:
                self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
                self.rnn1 = nn.LSTM(self.layer_1_size, 2000, self.LSTM_layers_1, batch_first=True)

            elif self.number_of_layers == 3:
                self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
                self.rnn1 = nn.LSTM(self.layer_1_size, self.layer_2_size, self.LSTM_layers_1, batch_first=True)
                self.linear2 = nn.Linear(self.layer_2_size, 2000)

            elif self.number_of_layers == 4:
                self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
                self.rnn1 = nn.LSTM(self.layer_1_size, self.layer_2_size, self.LSTM_layers_1, batch_first=True)
                self.linear2 = nn.Linear(self.layer_2_size, self.layer_3_size)
                self.rnn2 = nn.LSTM(self.layer_3_size, self.layer_4_size, self.LSTM_layers_2, batch_first=True)
                self.linear3 = nn.Linear(self.layer_4_size, 2000)

    def forward(self, x , y):

        if self.state_instead_input == True:

            if self.number_of_layers == 2:
                x = x.unsqueeze(dim=1)
                x = self.linear1(x)
                x_zeros = torch.zeros(x.shape)
                h = x.permute(1, 0, 2)
                c = torch.zeros(h.shape)
                x, _ = self.rnn1(x_zeros, (h, c))
                x = self.linear2(x)

            else:
                x = x.unsqueeze(dim=1)
                x = self.linear1(x)

                x_zeros = torch.zeros(x.shape)
                h = x.permute(1, 0, 2)
                c = torch.zeros(h.shape)
                x, _ = self.rnn1(x_zeros, (h, c))
                x = self.linear2(x)

                x_zeros = torch.zeros(x.shape)
                h = x.permute(1, 0, 2)
                c = torch.zeros(h.shape)
                x, _ = self.rnn2(x_zeros, (h, c))
                x = self.linear3(x)

        else:
            if self.number_of_layers == 2:
                x = x.unsqueeze(dim=1)
                x = self.linear1(x)
                x, _ = self.rnn1(x)

            elif self.number_of_layers == 3:
                x = x.unsqueeze(dim=1)
                x = self.linear1(x)
                x, _ = self.rnn1(x)
                x = self.linear2(x)

            elif self.number_of_layers == 4:
                x = x.unsqueeze(dim=1)
                x = self.linear1(x)
                x, _ = self.rnn1(x)
                x = self.linear2(x)
                x, _ = self.rnn2(x)
                x = self.linear3(x)

        return x


class GPT2(nn.Module):
    def __init__(self, config, number_classes, data, seq_length):
        super(GPT2, self).__init__()

        import pytorch_lightning as pl
        from pl_bolts.models.vision import ImageGPT
        # self.decoder = ImageGPT()

        # from pl_bolts.models.vision import PixelCNN
        # model = PixelCNN(input_channels=3)
        # x = torch.rand(5, 3, 64, 64)
        # t = model(x)

    def forward(self, x , y):
        '''
        x = self.linear1(x)
        h0 = torch.randn(2, 3, 20)
        c0 = torch.randn(2, 3, 20)
        #x = self.rnn(input, (h0, c0))
        x, _ = self.rnn1(x) #input(N (batchsize), L (seq. length), H_in (input_size))
        #x, _ = self.rnn2(x) #input(N (batchsize), L (seq. length), H_in (input_size))
        x = self.linear2(x)
        '''
        return self.decoder


class Resnet(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_encoder):
        super(Resnet, self).__init__()

        first_conv: bool = config["first_conv"]
        maxpool1: bool = config["maxpool1"]
        self.latent_space = config["latent_space"]
        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.final_layer_mu = nn.Linear(512, self.latent_space)
        self.final_layer_var = nn.Linear(512, self.latent_space)
        self.softmax = nn.LogSoftmax()

    def forward(self, x , y):
        x = self.encoder(x)
        x_mu = self.final_layer_mu(x)
        x_var = self.final_layer_var(x)
        #x = self.softmax(x)
        return x_mu, x_var

class Resnet(nn.Module):
    def __init__(self, config, number_classes, data):
        super(Resnet, self).__init__()

        first_conv: bool = False
        maxpool1: bool = False
        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.final_layer = nn.Linear(512, number_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.encoder(x)
        x = self.final_layer(x)
        x = self.softmax(x)
        return x

##### Resnet 1d

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class EncoderBlock(nn.Module):
    """
    ResNet block, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Resnet1d(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_encoder):
        super(Resnet1d, self).__init__()

        layers = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.conv1 = nn.Conv1d(4, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv1 = nn.Conv1d(4, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.latent_space = config["latent_space"]


        block = EncoderBlock
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        #self.final_layer = nn.Linear(1024, self.latent_space)
        #self.final_layer_var = nn.Linear(1024, self.latent_space)
        self.final_layer = nn.Linear(512, self.latent_space)
        self.final_layer_var = nn.Linear(512, self.latent_space)
        self.softmax = nn.LogSoftmax()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, y):
        x = x.view(x.shape[0], 4, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_mu = self.final_layer(x)
        x_var = self.final_layer_var(x)

        #x = self.softmax(x)

        return x_mu, x_var

class Resnet(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_encoder):
        super(Resnet, self).__init__()

        first_conv: bool = config["first_conv"]
        maxpool1: bool = config["maxpool1"]
        self.latent_space = config["latent_space"]
        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.final_layer_mu = nn.Linear(512, self.latent_space)
        self.final_layer_var = nn.Linear(512, self.latent_space)
        self.softmax = nn.LogSoftmax()

    def forward(self, x , y):
        x = self.encoder(x)
        x_mu = self.final_layer_mu(x)
        x_var = self.final_layer_var(x)
        #x = self.softmax(x)
        return x_mu, x_var

class Resnet(nn.Module):
    def __init__(self, config, number_classes, data):
        super(Resnet, self).__init__()

        first_conv: bool = False
        maxpool1: bool = False
        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.final_layer = nn.Linear(512, number_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.encoder(x)
        x = self.final_layer(x)
        x = self.softmax(x)
        return x

##### Resnet 1d

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class EncoderBlock(nn.Module):
    """
    ResNet block, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Resnet_dense(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_encoder):
        super(Resnet_dense, self).__init__()

        layers = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.conv1 = nn.Conv1d(4, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv1 = nn.Conv1d(4, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.latent_space = config["latent_space"]


        block = EncoderBlock
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.layer5 = self._make_layer(block, 1024, layers[4], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        #self.final_layer = nn.Linear(1024, self.latent_space)
        #self.final_layer_var = nn.Linear(1024, self.latent_space)
        self.final_layer = nn.Linear(512, self.latent_space)
        self.final_layer_var = nn.Linear(512, self.latent_space)
        self.softmax = nn.LogSoftmax()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, y):
        x = x.view(x.shape[0], 4, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_mu = self.final_layer(x)
        x_var = self.final_layer_var(x)

        #x = self.softmax(x)

        return x_mu, x_var

class Deep_single(pl.LightningModule): # original nn.module
    def __init__(self, config, number_classes, data, seq_length, y_encoder):
        global final_layer_size
        super(Deep_single, self).__init__()

        # clean if https://medium.com/swlh/3-alternatives-to-if-statements-to-make-your-python-code-more-readable-91a9991fb353
        self.number_classes = number_classes
        self.layer_1_size = 16
        self.layer_2_size = 512
        self.relu_on = 4
        self.layer_1_bias = False
        self.layer_2_bias = False
        self.layer_3_bias = False
        self.drop_0_n = 0.1
        self.drop_1_n = 0.1
        self.latent_space = 1
        first_layer_size = seq_length

        self.layer_0_1 = nn.Linear(first_layer_size, self.layer_1_size)
        self.batch_norm_0_1 = nn.BatchNorm1d(self.layer_1_size)
        self.layer_1_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
        self.batch_norm_1_1 = nn.BatchNorm1d(self.layer_2_size)
        self.layer_2_mu_1 = nn.Linear(self.layer_2_size, self.latent_space)
        self.layer_2_var_1 = nn.Linear(self.layer_2_size, self.latent_space)
        self.drop_0_1 = nn.Dropout(self.drop_0_n)
        self.drop_1_1 = nn.Dropout(self.drop_1_n)

        self.layer_0_2 = nn.Linear(first_layer_size, self.layer_1_size)
        self.batch_norm_0_2 = nn.BatchNorm1d(self.layer_1_size)
        self.layer_1_2 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
        self.batch_norm_1_2 = nn.BatchNorm1d(self.layer_2_size)
        self.layer_2_mu_2 = nn.Linear(self.layer_2_size, self.latent_space)
        self.layer_2_var_2 = nn.Linear(self.layer_2_size, self.latent_space)
        self.drop_0_2 = nn.Dropout(self.drop_0_n)
        self.drop_1_2 = nn.Dropout(self.drop_1_n)

        self.layer_0_3 = nn.Linear(first_layer_size, self.layer_1_size)
        self.batch_norm_0_3 = nn.BatchNorm1d(self.layer_1_size)
        self.layer_1_3 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
        self.batch_norm_1_3 = nn.BatchNorm1d(self.layer_2_size)
        self.layer_2_mu_3 = nn.Linear(self.layer_2_size, self.latent_space)
        self.layer_2_var_3 = nn.Linear(self.layer_2_size, self.latent_space)
        self.drop_0_3 = nn.Dropout(self.drop_0_n)
        self.drop_1_3 = nn.Dropout(self.drop_1_n)

        self.layer_0_4 = nn.Linear(first_layer_size, self.layer_1_size)
        self.batch_norm_0_4 = nn.BatchNorm1d(self.layer_1_size)
        self.layer_1_4 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
        self.batch_norm_1_4 = nn.BatchNorm1d(self.layer_2_size)
        self.layer_2_mu_4 = nn.Linear(self.layer_2_size, self.latent_space)
        self.layer_2_var_4 = nn.Linear(self.layer_2_size, self.latent_space)
        self.drop_0_4 = nn.Dropout(self.drop_0_n)
        self.drop_1_4 = nn.Dropout(self.drop_1_n)

        self.activation = torch.nn.Tanh()


    def forward(self, x):

        x1 = x[:, 0,:]
        x2 = x[:, 1, :]
        x3 = x[:, 2, :]
        x4 = x[:, 3, :]

        x1 = self.layer_0_1(x1)
        x1 = self.batch_norm_0_1(x1)
        x1 = self.activation(x1)
        x1 = self.drop_0_1(x1)
        x1 = self.layer_1_1(x1)
        x1 = self.batch_norm_1_1(x1)
        x1 = self.activation(x1)
        x1 = self.drop_1_1(x1)
        x_mu_1 = self.layer_2_mu_1(x1)
        x_var_1 = self.layer_2_var_1(x1)

        x2 = self.layer_0_2(x2)
        x2 = self.batch_norm_0_2(x2)
        x2 = self.activation(x2)
        x2 = self.drop_0_2(x2)
        x2 = self.layer_1_2(x2)
        x2 = self.batch_norm_1_2(x2)
        x2 = self.activation(x2)
        x2 = self.drop_1_2(x2)
        x_mu_2 = self.layer_2_mu_2(x2)
        x_var_2 = self.layer_2_var_2(x2)

        x3 = self.layer_0_3(x3)
        x3 = self.batch_norm_0_3(x3)
        x3 = self.activation(x3)
        x3 = self.drop_0_3(x3)
        x3 = self.layer_1_3(x3)
        x3 = self.batch_norm_1_3(x3)
        x3 = self.activation(x3)
        x3 = self.drop_1_3(x3)
        x_mu_3 = self.layer_2_mu_3(x3)
        x_var_3 = self.layer_2_var_3(x3)

        x4 = self.layer_0_4(x4)
        x4 = self.batch_norm_0_4(x4)
        x4 = self.activation(x4)
        x4 = self.drop_0_4(x4)
        x4 = self.layer_1_4(x4)
        x4 = self.batch_norm_1_4(x4)
        x4 = self.activation(x4)
        x4 = self.drop_1_4(x4)
        x_mu_4 = self.layer_2_mu_4(x4)
        x_var_4 = self.layer_2_var_4(x4)

        x_mu = torch.cat([x_mu_1, x_mu_2, x_mu_3, x_mu_4], dim=1)
        x_var = torch.cat([x_var_1, x_var_2, x_var_3, x_var_4], dim=1)

        return x_mu, x_var
