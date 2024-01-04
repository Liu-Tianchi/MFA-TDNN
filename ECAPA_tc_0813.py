"""
MFA-TDNN (multi-scale frequency-channel attention-TDNN)

@INPROCEEDINGS{9747021,
  author={Liu, Tianchi and Das, Rohan Kumar and Aik Lee, Kong and Li, Haizhou},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={{MFA: TDNN} with Multi-Scale Frequency-Channel Attention for Text-Independent Speaker Verification with Short Utterances}, 
  year={2022},
  volume={},
  number={},
  pages={7517-7521},
  doi={10.1109/ICASSP43922.2022.9747021}}

This script is based on the ECAPA-TDNN model @ SpeechBrain. Original author of ECAPA-TDNN script is:
 * Hwidong Na 2020
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear


# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class TDNNBlock(nn.Module):
    """An implementation of TDNN.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the TDNN blocks.
    dilation : int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class Conv2D_Basic_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=(0,0),
    ):
        super(Conv2D_Basic_Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = Res2NetBlock(64, 64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, out_channels, scale=8, dilation=1, dtype='TDNN'):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0
        self.dtype = dtype
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        if self.dtype == 'Conv2D':
            self.blocks = nn.ModuleList(
                [
                    Conv2D_Basic_Block(
                        in_channel, hidden_channel, kernel_size=(3,3), padding=(1,1), stride=(1,1)
                    )
                    for i in range(scale - 1)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TDNNBlock(
                        in_channel, hidden_channel, kernel_size=3, dilation=dilation
                    )
                    for i in range(scale - 1)
                ]
            )
        self.scale = scale
    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            if self.dtype == 'TDNN':
                y.append(y_i)
            else:
                y.append(torch.flatten(y_i, start_dim=1, end_dim=2))
        if self.dtype == 'TDNN':
            y = torch.cat(y, dim=1)
        return y

class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> se_layer = SEBlock(64, 16, 64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = se_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> x = torch.rand(8, 120, 64).transpose(1, 2)
    >>> conv = SERes2NetBlock(64, 64, res2net_scale=4)
    >>> out = conv(x).transpose(1, 2)
    >>> out.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
    ):
        super().__init__()
        self.out_channels = out_channels


        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, dilation, dtype='TDNN'
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)
        return x + residual

class Freq_att_layer(torch.nn.Module):
    def __init__(
            self,
            c_1st_conv,
            input_res2net_scale,
            SE_neur=8,
            sub_channel=160
    ):

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.att_se = nn.Sequential(
                nn.Linear(sub_channel, SE_neur),
                nn.ReLU(inplace=True),
                nn.Linear(SE_neur, sub_channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        # x = x.transpose(1,2)
        y = self.avg_pool(x).squeeze()
        y = self.att_se(y)
        y = y.unsqueeze(dim=-1)
        # y = y.unsqueeze(dim=-1)
        return x * y

class Freq_att_Block(torch.nn.Module):
    def __init__(
            self,
            c_1st_conv,
            outchannel,
            input_res2net_scale,
            SE_neur=8,
            se_channel=32,
            last_layer=False,
            res=True,

    ):
        super().__init__()
        self.last_layer = last_layer
        self.res = res
        assert (outchannel%input_res2net_scale)==0, 'in attention part, the outchannel%input_res2net_scale!=0'
        self.sub_channel = outchannel//input_res2net_scale
        print(self.sub_channel, outchannel, input_res2net_scale)
        self.blocks_att = nn.ModuleList(
            [
                Freq_att_layer(c_1st_conv, input_res2net_scale, SE_neur, self.sub_channel
                )
                for i in range(input_res2net_scale)
            ]
        )
        self.blocks_TDNN = nn.ModuleList(
            [
                TDNNBlock(
                    self.sub_channel, self.sub_channel, kernel_size=3, dilation=1
                )
                for i in range(input_res2net_scale)
            ]
        )
        if self.last_layer:
            self.conv1D = nn.Conv1d(outchannel, outchannel, kernel_size=1)
        self.se_block = SEBlock(outchannel, se_channel, outchannel)
    def forward(self, x):
        y = []

        for i in range(len(x)):
            if i == 0:
                y_i = self.blocks_att[i](x[i])
            else:
                y_i = self.blocks_att[i](x[i]+y_i)
            y_i = self.blocks_TDNN[i](y_i)
            y.append(y_i)
        if self.last_layer:
            y = torch.cat(y, dim=1)
            y = self.conv1D(y)
            if self.res:
                x = torch.cat(x, dim=1)
                y = y + x # res
        else:
            if self.res:
                for i in range(len(x)):
                    y[i] = y[i] + x[i]  # res
            y = torch.cat(y, dim=1)
        return y


class ECAPA_tc_0813(torch.nn.Module):
    
    '''
    MFA TDNN
    '''
    def __init__(
        self,
        input_size,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[640, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        c_1st_conv=32, # expend channel at the 1st step
        input_res2net_scale=4,  # feature extract in different scales at the 2nd step
        att_channel=640
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.att_channel = att_channel
        # network
        self.first_conv = nn.Sequential(
            Conv2D_Basic_Block(in_channels=1, out_channels=c_1st_conv, kernel_size=(3,3), padding=(1,1), stride=(2,1)),
            Conv2D_Basic_Block(in_channels=c_1st_conv, out_channels=c_1st_conv, kernel_size=(3, 3), padding=(1, 1), stride=(2, 1))
        ) # first step. expend feature channels
        self.res2conv2D = nn.Sequential(
            Res2NetBlock(c_1st_conv, c_1st_conv, scale=input_res2net_scale, dtype='Conv2D'),
        )# 2nd step. Multi-scale feature extraction
        self.freq_att_TDNN = nn.Sequential(
            Freq_att_Block(
                c_1st_conv,
                outchannel=self.att_channel,
                input_res2net_scale=input_res2net_scale,
                SE_neur=int(20*c_1st_conv/input_res2net_scale//5),
                se_channel=se_channels,
                last_layer=True,
                res=True),

        )# 3rd step. Att-TDNN

        # ----------------
        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            print(i,channels[i - 1],channels[i])
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,

                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)  # bz*6 80 301
        x = x.unsqueeze(dim=1) # bz*6 1 80 301
        x = self.first_conv(x) # bz*6 c_1st 40 301
        x = self.res2conv2D(x) # list: bz*6 c_1st/scale*40 301
        x = self.freq_att_TDNN(x) # list: bz*6 c_1st/scale*40 301
        # *6 indicates augmentations

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[0:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        x = x.transpose(1, 2)
        return x


class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=192,
        out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)
