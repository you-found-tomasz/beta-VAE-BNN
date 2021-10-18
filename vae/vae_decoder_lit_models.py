import torch
from torch import nn
import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)


def load_model(config, network, number_classes, data, seq_length, y_decoder, single_to_decoder):
    if network == "deep":
        selected_network = Deep(config, number_classes, data, seq_length, y_decoder, single_to_decoder)
    elif network == "conv":
        selected_network = Conv(config, number_classes, data, seq_length, y_decoder)
    elif network == "LSTM":
        selected_network = LSTM(config, number_classes, data, seq_length, y_decoder)
    elif network == "Resnet":
        selected_network = Resnet(config, number_classes, data, seq_length, y_decoder)
    elif network == "GPT2":
        selected_network = GPT2(config, number_classes, data, seq_length, y_decoder)
    else:
        print("no network")
        selected_network = []

    return selected_network


class Deep(pl.LightningModule): #original working nn.Module
    def __init__(self, config, number_classes, data, seq_length, y_decoder, single_to_decoder):
        global final_layer_size
        super(Deep, self).__init__()

        # clean if https://medium.com/swlh/3-alternatives-to-if-statements-to-make-your-python-code-more-readable-91a9991fb353

        self.number_classes = number_classes
        self.number_of_layers = config["dec_number_of_layers"]
        self.layer_1_size = config["dec_layer_1_size"]
        self.layer_2_size = config["dec_layer_2_size"]
        self.layer_3_size = config["dec_layer_3_size"]
        self.layer_4_size = config["dec_layer_4_size"]
        self.layer_5_size = config["dec_layer_5_size"]
        self.layer_6_size = config["dec_layer_6_size"]
        self.layer_7_size = config["dec_layer_7_size"]
        self.relu_on = config["dec_activation"]
        self.layer_1_bias = config["dec_layer_1_bias"]
        self.layer_2_bias = config["dec_layer_2_bias"]
        self.layer_3_bias = config["dec_layer_3_bias"]
        self.layer_4_bias = config["dec_layer_4_bias"]
        self.layer_5_bias = config["dec_layer_5_bias"]
        self.layer_6_bias = config["dec_layer_6_bias"]
        self.drop_0_n = config["dec_drop_0"]
        self.drop_1_n = config["dec_drop_1"]
        self.drop_2_n = config["dec_drop_2"]
        self.drop_3_n = config["dec_drop_3"]
        self.drop_4_n = config["dec_drop_4"]
        if single_to_decoder == "False":
            self.latent_space = config["latent_space"] + 4 #if
        else:
            self.latent_space = config["latent_space"]
        self.y_decoder = y_decoder

        final_layer_size = seq_length*4

        if self.number_of_layers == 2:
            if self.y_decoder == True:
                self.layer_0 = nn.Linear(self.latent_space + 15, self.layer_1_size)
            else:
                self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, final_layer_size, bias=self.layer_1_bias)

        elif self.number_of_layers == 3:
            if self.y_decoder == True:
                self.layer_0 = nn.Linear(self.latent_space + 15, self.layer_1_size)
            else:
                self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
            self.batch_norm_1 = nn.BatchNorm1d(self.layer_2_size)
            self.layer_2 = nn.Linear(self.layer_2_size, final_layer_size)

        elif self.number_of_layers == 4:
            if self.y_decoder == True:
                self.layer_0 = nn.Linear(self.latent_space + 15, self.layer_1_size)
            else:
                self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
            self.batch_norm_1 = nn.BatchNorm1d(self.layer_2_size)
            self.layer_2 = nn.Linear(self.layer_2_size, self.layer_3_size, bias=self.layer_2_bias)
            self.batch_norm_2 = nn.BatchNorm1d(self.layer_3_size)
            self.layer_3 = nn.Linear(self.layer_3_size, final_layer_size)
            self.drop_3 = nn.Dropout(self.drop_3_n)

        elif self.number_of_layers == 5:
            if self.y_decoder == True:
                self.layer_0 = nn.Linear(self.latent_space + 15, self.layer_1_size)
            else:
                self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
            self.batch_norm_1 = nn.BatchNorm1d(self.layer_2_size)
            self.layer_2 = nn.Linear(self.layer_2_size, self.layer_3_size, bias=self.layer_2_bias)
            self.batch_norm_2 = nn.BatchNorm1d(self.layer_3_size)
            self.layer_3 = nn.Linear(self.layer_3_size, self.layer_4_size, bias=self.layer_3_bias)
            self.batch_norm_3 = nn.BatchNorm1d(self.layer_4_size)
            self.layer_4 = nn.Linear(self.layer_4_size, final_layer_size)

        elif self.number_of_layers == 6:
            if self.y_decoder == True:
                self.layer_0 = nn.Linear(self.latent_space + 15, self.layer_1_size)
            else:
                self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.layer_0 = nn.Linear(self.latent_space, self.layer_1_size)
            self.batch_norm_0 = nn.BatchNorm1d(self.layer_1_size)
            self.layer_1 = nn.Linear(self.layer_1_size, self.layer_2_size, bias=self.layer_1_bias)
            self.batch_norm_1 = nn.BatchNorm1d(self.layer_2_size)
            self.layer_2 = nn.Linear(self.layer_2_size, self.layer_3_size, bias=self.layer_2_bias)
            self.batch_norm_2 = nn.BatchNorm1d(self.layer_3_size)
            self.layer_3 = nn.Linear(self.layer_3_size, self.layer_4_size, bias=self.layer_3_bias)
            self.batch_norm_3 = nn.BatchNorm1d(self.layer_4_size)
            self.layer_4 = nn.Linear(self.layer_4_size, self.layer_5_size, bias=self.layer_4_bias)
            self.layer_5 = nn.Linear(self.layer_5_size, final_layer_size)

        self.activation = torch.nn.ReLU() if self.relu_on == "relu" else torch.nn.Tanh()
        self.drop_0 = nn.Dropout(self.drop_0_n)
        self.drop_1 = nn.Dropout(self.drop_1_n)
        self.drop_2 = nn.Dropout(self.drop_2_n)
        self.drop_3 = nn.Dropout(self.drop_3_n)
        self.drop_4 = nn.Dropout(self.drop_4_n)

    def forward(self, x , y):
        if self.number_of_layers == 2:
            if self.y_decoder == True:
                x = torch.cat([x, y], dim=1)
            x = self.layer_0(x)
            x = self.batch_norm_0(x)
            x = self.activation(x)
            x = self.drop_0(x)
            x = self.layer_1(x)

        elif self.number_of_layers == 3:
            if self.y_decoder == True:
                x = torch.cat([x, y], dim=1)
            x = self.layer_0(x)
            x = self.batch_norm_0(x)
            x = self.activation(x)
            x = self.drop_0(x)
            x = self.layer_1(x)
            x = self.batch_norm_1(x)
            x = self.activation(x)
            x = self.drop_1(x)
            x = self.layer_2(x)

        elif self.number_of_layers == 4:
            if self.y_decoder == True:
                x = torch.cat([x, y], dim=1)
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

        elif self.number_of_layers == 5:
            if self.y_decoder == True:
                x = torch.cat([x, y], dim=1)
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

        elif self.number_of_layers == 6:
            if self.y_decoder == True:
                x = torch.cat([x, y], dim=1)
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

            x = self.layer_5(x)

        return x


class Conv(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_decoder):
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
        self.bias_1_size = config["bias_1_size"]
        self.bias_2_size = config["bias_2_size"]
        self.bias_3_size = config["bias_3_size"]
        self.bias_4_size = config["bias_4_size"]
        self.bias_5_size = config["bias_5_size"]
        if data == "raw" or data == "raw_jit":
            self.final_layer_size = 500
        elif data == "raw_2000":
            self.final_layer_size = 2000
        elif data == "raw_4000":
            self.final_layer_size = 4000

        if self.number_of_layers == 2:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=4,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=self.bias_1_size)

        elif self.number_of_layers == 3:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=self.bias_1_size)
            self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=4,
                                              kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0,
                                              bias=self.bias_2_size)

        elif self.number_of_layers == 4:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=self.bias_1_size)
            self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=self.channel_3_size,
                                              kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0,
                                              bias=self.bias_2_size)
            self.deconv3 = nn.ConvTranspose1d(in_channels=self.channel_3_size, out_channels=4,
                                              kernel_size=self.kernel_3_size, stride=self.stride_3_size, padding=0,
                                              bias=self.bias_3_size)

        elif self.number_of_layers == 5:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=self.bias_1_size)
            self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=self.channel_3_size,
                                              kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0,
                                              bias=self.bias_2_size)
            self.deconv3 = nn.ConvTranspose1d(in_channels=self.channel_3_size, out_channels=self.channel_4_size,
                                              kernel_size=self.kernel_3_size, stride=self.stride_3_size, padding=0,
                                              bias=self.bias_3_size)
            self.deconv4 = nn.ConvTranspose1d(in_channels=self.channel_4_size, out_channels=4,
                                              kernel_size=self.kernel_4_size, stride=self.stride_4_size, padding=0,
                                              bias=self.bias_4_size)

        elif self.number_of_layers == 6:
            self.linear1 = nn.Linear(number_classes, self.channel_1_size)
            self.deconv1 = nn.ConvTranspose1d(in_channels=self.channel_1_size, out_channels=self.channel_2_size,
                                              kernel_size=self.kernel_1_size, stride=self.stride_1_size, padding=0,
                                              bias=self.bias_1_size)
            self.deconv2 = nn.ConvTranspose1d(in_channels=self.channel_2_size, out_channels=self.channel_3_size,
                                              kernel_size=self.kernel_2_size, stride=self.stride_2_size, padding=0,
                                              bias=self.bias_2_size)
            self.deconv3 = nn.ConvTranspose1d(in_channels=self.channel_3_size, out_channels=self.channel_4_size,
                                              kernel_size=self.kernel_3_size, stride=self.stride_3_size, padding=0,
                                              bias=self.bias_3_size)
            self.deconv4 = nn.ConvTranspose1d(in_channels=self.channel_4_size, out_channels=self.channel_5_size,
                                              kernel_size=self.kernel_4_size, stride=self.stride_4_size, padding=0,
                                              bias=self.bias_4_size)
            self.deconv5 = nn.ConvTranspose1d(in_channels=self.channel_5_size, out_channels=4,
                                              kernel_size=self.kernel_5_size, stride=self.stride_5_size, padding=0,
                                              bias=self.bias_5_size)


    def forward(self, x, y):

        if self.number_of_layers == 2:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = torch.nn.functional.interpolate(x, size=self.final_layer_size)

        elif self.number_of_layers == 3:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = torch.nn.functional.interpolate(x, size=self.final_layer_size)

        elif self.number_of_layers == 4:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = torch.nn.functional.interpolate(x, size=self.final_layer_size)

        elif self.number_of_layers == 5:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
            x = torch.nn.functional.interpolate(x, size=self.final_layer_size)

        elif self.number_of_layers == 6:
            x = self.linear1(x)
            x = x.view(-1, self.channel_1_size, 1)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
            x = self.deconv5(x)
            x = torch.nn.functional.interpolate(x, size=self.final_layer_size)

        return x


class LSTM(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_decoder):
        super(LSTM, self).__init__()

        self.number_classes = number_classes
        self.number_of_layers = config["number_of_layers"]

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.layer_3_size = config["layer_3_size"]
        self.layer_4_size = config["layer_4_size"]

        self.LSTM_layers_1 = config["LSTM_layers_1"]
        self.LSTM_layers_2 = config["LSTM_layers_2"]

        final_layer_size = 0
        if data == "raw" or data == "raw_jit":
            final_layer_size = 2000
        elif data == "raw_2000":
            final_layer_size = 8000
        elif data == "raw_4000":
            final_layer_size = 16000

        if self.number_of_layers == 2:
            self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
            self.rnn1 = nn.LSTM(self.layer_1_size, self.layer_1_size, 1, batch_first=True)
            self.linear2 = nn.Linear(self.layer_1_size, final_layer_size)

        elif self.number_of_layers == 3:
            self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
            self.rnn1 = nn.LSTM(self.layer_1_size, self.layer_1_size, 1, batch_first=True)
            self.linear2 = nn.Linear(self.layer_1_size, self.layer_2_size)
            self.rnn2 = nn.LSTM(self.layer_2_size, self.layer_2_size, 1, batch_first=True)
            self.linear3 = nn.Linear(self.layer_2_size, final_layer_size)

        elif self.number_of_layers == 4:
            self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
            self.rnn1 = nn.LSTM(self.layer_1_size, self.layer_2_size, self.LSTM_layers_1, batch_first=True)
            self.linear2 = nn.Linear(self.layer_2_size, final_layer_size)

        elif self.number_of_layers == 5:
            self.linear1 = nn.Linear(self.number_classes, self.layer_1_size)
            self.rnn1 = nn.LSTM(self.layer_1_size, self.layer_2_size, self.LSTM_layers_1, batch_first=True)
            self.linear2 = nn.Linear(self.layer_2_size, self.layer_3_size)
            self.rnn2 = nn.LSTM(self.layer_3_size, self.layer_4_size, self.LSTM_layers_2, batch_first=True)
            self.linear3 = nn.Linear(self.layer_4_size, final_layer_size)

    def forward(self, x, y):

        if self.number_of_layers == 2:
            x = x.unsqueeze(dim=1)
            x = self.linear1(x)
            x_zeros = torch.zeros(x.shape)
            h = x.permute(1, 0, 2)
            c = torch.zeros(h.shape)
            x, _ = self.rnn1(x_zeros, (h, c))
            x = self.linear2(x)

        elif self.number_of_layers == 3:
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

        elif self.number_of_layers == 4:
            x = x.unsqueeze(dim=1)
            x = self.linear1(x)
            x, _ = self.rnn1(x)
            x = self.linear2(x)

        elif self.number_of_layers == 5:
            x = x.unsqueeze(dim=1)
            x = self.linear1(x)
            x, _ = self.rnn1(x)
            x = self.linear2(x)
            x, _ = self.rnn2(x)
            x = self.linear3(x)

        return x


class GPT2(nn.Module):
    def __init__(self, config, number_classes, data, seq_length, y_decoder):
        super(GPT2, self).__init__()

        import pytorch_lightning as pl
        from pl_bolts.models.vision import ImageGPT
        # self.decoder = ImageGPT()

        # from pl_bolts.models.vision import PixelCNN
        # model = PixelCNN(input_channels=3)
        # x = torch.rand(5, 3, 64, 64)
        # t = model(x)

    def forward(self, x, y):
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
    def __init__(self, config, number_classes, data, seq_length, y_decoder):
        super(Resnet, self).__init__()

        input_height: int = 256
        first_conv: bool = False
        maxpool1: bool = False
        latent_dim: int = number_classes
        self.decoder = resnet18_decoder(latent_dim, input_height, first_conv, maxpool1)

    def forward(self, x, y):
        return self.decoder(x)

class Deep_single(pl.LightningModule): #original working nn.Module
    def __init__(self, config, number_classes, data, seq_length, y_decoder):
        global final_layer_size
        super(Deep_single, self).__init__()

        # clean if https://medium.com/swlh/3-alternatives-to-if-statements-to-make-your-python-code-more-readable-91a9991fb353

        self.number_classes = number_classes
        self.number_of_layers = 2
        self.layer_1_size = 512
        self.layer_1_bias = True
        self.drop_0_n = 0
        self.latent_space = config["latent_space"]
        self.y_decoder = y_decoder

        final_layer_size = seq_length

        self.layer_0_1 = nn.Linear(1, self.layer_1_size)
        self.batch_norm_0_1 = nn.BatchNorm1d(self.layer_1_size)
        self.layer_1_1 = nn.Linear(self.layer_1_size, final_layer_size, bias=self.layer_1_bias)
        self.drop_0_1 = nn.Dropout(self.drop_0_n)

        self.layer_0_2 = nn.Linear(1, self.layer_1_size)
        self.batch_norm_0_2 = nn.BatchNorm1d(self.layer_1_size)
        self.layer_1_2 = nn.Linear(self.layer_1_size, final_layer_size, bias=self.layer_1_bias)
        self.drop_0_2 = nn.Dropout(self.drop_0_n)

        self.layer_0_3 = nn.Linear(1, self.layer_1_size)
        self.batch_norm_0_3 = nn.BatchNorm1d(self.layer_1_size)
        self.layer_1_3 = nn.Linear(self.layer_1_size, final_layer_size, bias=self.layer_1_bias)
        self.drop_0_3 = nn.Dropout(self.drop_0_n)

        self.layer_0_4 = nn.Linear(1, self.layer_1_size)
        self.batch_norm_0_4 = nn.BatchNorm1d(self.layer_1_size)
        self.layer_1_4 = nn.Linear(self.layer_1_size, final_layer_size, bias=self.layer_1_bias)
        self.drop_0_4 = nn.Dropout(self.drop_0_n)

        self.activation = torch.nn.ReLU()


    def forward(self, x):

        x1 = x[:,-1].unsqueeze(1)
        x2 = x[:,-2].unsqueeze(1)
        x3 = x[:,-3].unsqueeze(1)
        x4 = x[:,-4].unsqueeze(1)


        x1 = self.layer_0_1(x1)
        x1 = self.batch_norm_0_1(x1)
        x1 = self.activation(x1)
        x1 = self.drop_0_1(x1)
        x1 = self.layer_1_1(x1)

        x2 = self.layer_0_2(x2)
        x2 = self.batch_norm_0_2(x2)
        x2 = self.activation(x2)
        x2 = self.drop_0_2(x2)
        x2 = self.layer_1_2(x2)

        x3 = self.layer_0_3(x3)
        x3 = self.batch_norm_0_3(x3)
        x3 = self.activation(x3)
        x3 = self.drop_0_3(x3)
        x3 = self.layer_1_3(x3)

        x4 = self.layer_0_4(x4)
        x4 = self.batch_norm_0_4(x4)
        x4 = self.activation(x4)
        x4 = self.drop_0_4(x4)
        x4 = self.layer_1_4(x4)

        x = torch.cat([x1,x2,x3,x4],dim=1)

        return x
