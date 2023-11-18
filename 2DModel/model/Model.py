import torch
import torch.nn as nn
import importlib
from Config_doc.logger import get_logger

logger = get_logger('Model')

class PreNet(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(PreNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # logger.info(f'PreNet first_in: {in_channels}')
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1))
        in_channels *= 2
        # print('first_out', in_channels)

        # Encoder
        for _ in range(6):
            # print('en_in', in_channels)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(in_channels * 2)
            ))
            in_channels *= 2
            # print('en_out', in_channels)

        # Decoder
        for _ in range(6):
            # print('de_in', in_channels)
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels * 2, in_channels // 2, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(in_channels // 2)
            ))
            in_channels //= 2
            # print('de_out', in_channels)

        # print('fil_in', in_channels)

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        # print('fil', out_channels)

    def forward(self, x):
        encoder_outs = []
        x = self.first_conv(x)
        # print('first_out', x.shape)
        encoder_outs.append(x)

        # Encoder forward pass
        for encoder_block in self.encoder:
            # print('en_in', x.shape)
            x = encoder_block(x)
            encoder_outs.append(x)
#             print('en_out', x.shape)
        # print(encoder_outs)
        # Decoder forward pass
        for decoder_block in self.decoder:
#             print('de_in', x.shape)
            en_x = encoder_outs.pop()
            # print('en_pop', en_x.shape)
            x = torch.cat([x, en_x], dim=1)
            # print('ca_out', x.shape)
            x = decoder_block(x)
#             print('de_out', x.shape)
        x = self.final_conv(x)
#         print('final',x.shape)

        return x

class ARNet_or(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(ARNet_or, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU for encoder
        )

        self.maxp = nn.MaxPool2d((2, 2), 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)  # ReLU for decoder
        )

        self.contr = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride = 1, padding = 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder forward pass
        # for encoder_block in self.encoder:

        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.maxp(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.maxp(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        # print('decode')
        x = self.decoder(x)
        # print(x.shape)
        x = self.contr(x)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        x = self.contr(x)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        x = self.final_conv(x)
        # print(x.shape)
        rectified_params = self.tanh(x)
        # logger.info(f'ARNet Output:{rectified_params.shape}')

        return rectified_params



class ARNet(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(ARNet, self).__init__()
        # self.encoder = nn.ModuleList()
        # self.decoder = nn.ModuleList()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU for encoder
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU for encoder
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU for encoder
        )

        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU for encoder
        )

        self.maxp = nn.MaxPool2d(2, 2)

        # Decoder
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)  # ReLU for decoder
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)  # ReLU for decoder
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(32,16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)  # ReLU for decoder
        )
        # for _ in range(3):
        #     self.decoder.append(nn.Sequential(
        #         nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=1),
        #         nn.BatchNorm2d(in_channels),
        #         nn.ReLU(inplace=True)  # ReLU for decoder
        #     ))

        self.contr_1 = nn.ConvTranspose2d(64,64, kernel_size=2, stride=2)
        self.contr_2 = nn.ConvTranspose2d(32,32, kernel_size = 2, stride = 2)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=3,stride = 1, padding = 1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder forward pass
        # for encoder_block in self.encoder:

        # print(x.shape)
        x = self.encoder_1(input)
        # print(x.shape)
        x = self.maxp(x)
        # print(x.shape)
        x = self.encoder_2(x)
        # print(x.shape)
        x = self.maxp(x)
        # print(x.shape)
        x = self.encoder_3(x)
        # print(x.shape)
        x = self.encoder_4(x)
        # print(x.shape)
        # print('decode')
        x = self.decoder_1(x)
        # print(x.shape)
        x = self.contr_1(x)
        # print(x.shape)
        x = self.decoder_2(x)
        # print(x.shape)
        x = self.contr_2(x)
        # print(x.shape)
        x = self.decoder_3(x)
        # print(x.shape)
        x = self.final_conv(x)
        # print(x.shape)
        last = self.tanh(x)
        # print(rectified_params.shape)
        rectified_params = torch.mul(input,last)

        return rectified_params

class AdvNet(nn.Module):
    def __init__(self,in_channels, out_channels,**kwargs):
        super(AdvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # logger.info(f'ADV Input:{x.shape}')
        x = self.conv_layers(x)
        x = self.output_layer(x)
        # logger.info(f'ADV Onput:{x.shape}')
        return x

def get_class(class_name, modules): # get the model or class 
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')

def get_model(model_config):
    model_class = get_class(model_config['name'], modules=['model.Model'])
    return model_class(**model_config)