import torch
import torch.nn as nn
import torch.nn.functional as F

class convLSTM_basic(nn.Module):
    def __init__(self, input_shape, in_channels, out_channel, kernel_size, stride):
        super(convLSTM_basic, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.sigma = nn.ReLU()

        # convolutions has to be stored in (out_channels, in_channels, kernel_size, kernel_size) format
        self.Wxi = nn.Parameter(torch.Tensor(self.out_channel, self.in_channels, self.kernel_size, self.kernel_size))
        self.Wxi_bias = nn.Parameter(torch.Tensor(self.out_channel))

        self.Whi = nn.Parameter(torch.Tensor(self.out_channel, self.out_channel, self.kernel_size, self.kernel_size))
        self.Whi_bias = nn.Parameter(torch.Tensor(self.out_channel))

        self.Wci = nn.Parameter(torch.Tensor(self.out_channel, input_shape[0], input_shape[1]))
        self.Wci_bias = nn.Parameter(torch.Tensor(self.out_channel))



        self.Wxf = nn.Parameter(torch.Tensor(self.out_channel, self.in_channels, self.kernel_size, self.kernel_size))
        self.Wxf_bias = nn.Parameter(torch.Tensor(self.out_channel))

        self.Whf = nn.Parameter(torch.Tensor(self.out_channel, self.out_channel, self.kernel_size, self.kernel_size))
        self.Whf_bias = nn.Parameter(torch.Tensor(self.out_channel))

        self.Wcf = nn.Parameter(torch.Tensor(self.out_channel, input_shape[0], input_shape[1]))
        self.Wcf_bias = nn.Parameter(torch.Tensor(self.out_channel))


        self.Wxc = nn.Parameter(torch.Tensor(self.out_channel, self.in_channels, self.kernel_size, self.kernel_size))
        self.Wxc_bias = nn.Parameter(torch.Tensor(self.out_channel))

        self.Whc = nn.Parameter(torch.Tensor(self.out_channel, self.out_channel, self.kernel_size, self.kernel_size))
        self.Whc_bias = nn.Parameter(torch.Tensor(self.out_channel))


        self.Wxo = nn.Parameter(torch.Tensor(self.out_channel, self.in_channels, self.kernel_size, self.kernel_size))
        self.Wxo_bias = nn.Parameter(torch.Tensor(self.out_channel))

        self.Who = nn.Parameter(torch.Tensor(self.out_channel, self.out_channel, self.kernel_size, self.kernel_size))
        self.Who_bias = nn.Parameter(torch.Tensor(self.out_channel))

        self.Wco = nn.Parameter(torch.Tensor(self.out_channel, input_shape[0], input_shape[1]))

        self.bi  = nn.Parameter(torch.Tensor(self.out_channel, input_shape[0], input_shape[1]))
        self.bf = nn.Parameter(torch.Tensor(self.out_channel, input_shape[0], input_shape[1]))
        self.bc = nn.Parameter(torch.Tensor(self.out_channel, input_shape[0], input_shape[1]))
        self.bo = nn.Parameter(torch.Tensor(self.out_channel, input_shape[0], input_shape[1]))

    def forward(self, input_tensor, cur_state):
        # equations for convLSTM as in original paper
        # Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
        # * is convolution, \hadamard is hadamard product
        # i_t = sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + W_{ci} \hadamard C_{t-1} + b_{i})
        # f_t = sigma(w_{xf} * X_t + W_{hf} * H_{t-1} + W_{cf} \hadamard C_{t-1} + b_{f})
        # C_{t} = f_{t} \hadamard C_{t-1} + i_{t} \hadamard tanh(W_{xc} * X_{t} + W_{hc} * H_{t-1} + b_{c})
        # o_{t} = \sigma(W_{xo} * X_{t} + W_{ho} * H_{t-1} + W_{co} \hadamard C_{t} + b_{o})
        # H_{t} = o_{t} \hadamard tanh(C_{t})

        # input tensor here is B x in_ch x W x H
        # hidden tensor is B x out


        h_current, c_current = cur_state

        #
        i_t = self.sigma(F.conv2d(input_tensor, self.Wxi, bias=self.Wxi_bias, stride=1, padding=1) +
                         F.conv2d(h_current, self.Whi, bias=self.Whi_bias, stride=1, padding=1) + self.Wci * c_current + self.bi)

        f_t = self.sigma(F.conv2d(input_tensor, self.Wxf, bias=self.Wxf_bias, stride=1, padding=1) +
                         F.conv2d(h_current, self.Whf, bias=self.Whf_bias, stride=1, padding=1) + self.Wcf * c_current + self.bf)

        c_next = f_t * c_current + i_t * F.tanh(F.conv2d(input_tensor, self.Wxc, bias=self.Wxc_bias, stride=1, padding=1) +
                                                F.conv2d(h_current, self.Whc, bias=self.Whc_bias, stride=1, padding=1) + self.bc)

        o_t = self.sigma(F.conv2d(input_tensor, self.Wxo, bias=self.Wxo_bias, stride=1, padding=1) +
                         F.conv2d(h_current, self.Who, bias=self.Who_bias, stride=1, padding=1) + self.Wco * c_current + self.bo)

        h_next = o_t * F.tanh(c_next)

        return h_next, c_next

class ConvLSTMCell(nn.Module):
    """
        Inpout : (B, in_channels, H, W)
        State: (h, c) each with (B, out_channels, H, W)
        output: same shape as state
    """
    def __init__(self, in_channels, out_channels, ks=3, bias=True, peephole=True):
        super().__init__()
        pad = ks // 2
        self.peephole = peephole
        self.conv_x = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=ks, padding=pad, bias=bias)
        self.conv_h = nn.Conv2d(out_channels, 4 * out_channels, kernel_size=ks, padding=pad, bias=bias)

        if peephole:
            self.Wci = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
            self.Wcf = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
            self.Wco = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, x, state):
        # previous state
        h, c = state

        # one convolution each -> split channels
        xc_i, xc_f, xc_g, xc_o =  self.conv_x(x).chunk(4, 1)
        hc_i, hc_f, hc_g, hc_o = self.conv_h(h).chunk(4, 1)

        if self.peephole:
            i = torch.sigmoid(xc_i + hc_i + self.Wci * c)
            f = torch.sigmoid(xc_f + hc_f + self.Wcf * c)
        else:
            i = torch.sigmoid(xc_i + hc_i)
            f = torch.sigmoid(xc_f + hc_f)

        g = torch.tanh(xc_g + hc_g)
        c_next = f * c + i * g

        if self.peephole:
            o = torch.sigmoid(xc_o + hc_o + self.Wco * c_next)
        else:
            o = torch.sigmoid(xc_o + hc_o)

        h_next = o * torch.tanh(c_next)
        return h_next, c_next

"""
if __name__ == "__main__":

    input_shape = (120, 80)
    channels = 5
    batch_size = 2
    out_channels = 3

    input_tensor = torch.randn((batch_size, channels) + input_shape)
    cur_state = (torch.zeros(batch_size, out_channels, input_shape[0], input_shape[1]),
                 torch.zeros(batch_size, out_channels, input_shape[0], input_shape[1]))

    model = convLSTM(input_shape, channels, 3, 3, 1)

    for i in range(5):
        next_state = model(input_tensor, cur_state)
        cur_state = next_state

    h_next, c_next = next_state
    print(h_next.shape)
    print(c_next.shape)
"""

if __name__ == "__main__":
    input_shape = (180, 240)
    channels = 5
    batch_size = 2
    out_channels = 3

    input_tensor = torch.randn((batch_size, channels) + input_shape)
    cur_state = (torch.zeros(batch_size, out_channels, input_shape[0], input_shape[1]),
                 torch.zeros(batch_size, out_channels, input_shape[0], input_shape[1]))

    model = ConvLSTMCell(in_channels=channels, out_channels=out_channels, ks=3, bias=True, peephole=True)
    print("input shape ", input_tensor.shape)
    print("current shape[0] ", cur_state[0].shape)
    print("current shape[1]", cur_state[1].shape)
    for i in range(5):
        next_state = model(input_tensor, cur_state)
        cur_state = next_state

    h_next, c_next = next_state
    print(h_next.shape)
    print(c_next.shape)

