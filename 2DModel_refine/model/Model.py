import torch
import torch.nn as nn
import importlib
from Config_doc.logger import get_logger
from torch.nn import functional as F
from torch.nn import init

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

class ARNet(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(ARNet, self).__init__()
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


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1)
            # upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up


        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            # print(self.model(x).size())
            return self.model(x)
        else:
            # return torch.cat([self.model(x), x], 1)
            output = self.model(x)
            size = x.size()[2:]
            middle_output = F.interpolate(output, size=size, mode='bilinear')
            return torch.cat([middle_output, x], 1)

class DilatedBlock(nn.Module):
    def __init__(self, input_nc, output_nc, dilatedRate, norm_layer, use_dropout):
        super(DilatedBlock, self).__init__()
        self.conv_block = self.build_Dilatedconv_block(input_nc, output_nc, dilatedRate, norm_layer, use_dropout)

    def build_Dilatedconv_block(self, input_nc, output_nc, dilatedRate, norm_layer, use_dropout):
        conv_block = []
        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=dilatedRate,
                       dilation = dilatedRate),
                       norm_layer(output_nc, affine=True),
                       nn.LeakyReLU(0.2, True)]
        # if use_dropout:
        #     conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = torch.cat([self.conv_block(x), x], 1)
        return out

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids


        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        # unet_block += [nn.Softmax(dim=1)]

        self.model = unet_block

    def forward(self, input):
        # print(input.size(), self.model(input).size())
        return self.model(input)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class DilatedNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(DilatedNet, self).__init__()
        self.gpu_ids = gpu_ids

        DilatedCount = 1
        inputCount = 0

        # construct unet structure
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=DilatedCount,
                           dilation=DilatedCount),
                 norm_layer(ngf, affine=True),
                 nn.LeakyReLU(0.2, True)]
        DilatedCount *= 2
        inputCount += ngf
        for i in range(num_downs - 2):
            model += [DilatedBlock(inputCount, ngf, DilatedCount, norm_layer, use_dropout)]
            DilatedCount *= 2
            inputCount += ngf

        model += [nn.Conv2d(inputCount, ngf, kernel_size=3, padding=1),
                  norm_layer(ngf, affine=True),
                  nn.LeakyReLU(0.2, True)]

        # model += [nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1),
        #           nn.Tanh()]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]

        # unet_block += [nn.Softmax(dim=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type=init_type)
    net.apply(init_weights(init_type))
    return net

def init_weights(net, init_type='normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=1)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)
    return init_func


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(in_channels, out_channels, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[],
                 **kwargs):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    input_nc = in_channels
    output_nc = out_channels

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                   gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                                   gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_5blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=5,
                                   gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_32':
        netG = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                 gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                 gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                 gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                 gpu_ids=gpu_ids)
    elif which_model_netG == 'dilated_32':
        netG = DilatedNet(input_nc, output_nc, 5, ngf=32, norm_layer=norm_layer, use_dropout=use_dropout,
                              gpu_ids=gpu_ids)
    elif which_model_netG == 'dilated_64':
        netG = DilatedNet(input_nc, output_nc, 6, ngf=32, norm_layer=norm_layer, use_dropout=use_dropout,
                              gpu_ids=gpu_ids)
    elif which_model_netG == 'dilated_128':
        netG = DilatedNet(input_nc, output_nc, 7, ngf=32, norm_layer=norm_layer, use_dropout=use_dropout,
                              gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)

    init_type = 'kaiming'

        ###### new version #############
    return init_net(netG, init_type, gpu_ids)
        # return netG


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