import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_size, kernel_size):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(in_channels + hidden_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(in_channels + hidden_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(in_channels + hidden_size, hidden_size, kernel_size=kernel_size, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)

        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, x, prev_state):

        # input shape is (B, in_channels, height, width)
        # state shape is (B, hidden_size, height, width)
        stacked_inputs = torch.cat([x, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))

        out_inputs = F.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))

        new_state = prev_state * (1.0 - update) + out_inputs * update

        return new_state

if __name__ == "__main__":
    input_shape = (180, 240)
    channels = 5
    batch_size = 2
    out_channels = 3

    input_tensor = torch.randn((batch_size, channels) + input_shape)
    cur_state = torch.zeros(batch_size, out_channels, input_shape[0], input_shape[1])

    model = ConvGRUCell(in_channels=channels, hidden_size=out_channels, kernel_size=3)
    print("input shape ", input_tensor.shape)
    print("current shape[0] ", cur_state[0].shape)
    print("current shape[1]", cur_state[1].shape)
    for i in range(5):
        next_state = model(input_tensor, cur_state)
        cur_state = next_state

    h_next, c_next = next_state
    print(h_next.shape)
    print(c_next.shape)
