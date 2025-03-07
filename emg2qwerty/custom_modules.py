import torch
from torch import nn


class BandedLSTM(nn.Module):
    """An LSTM that applies to spectrogram input from separated band channels."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm= nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, B, C, F = inputs.shape
        assert B * C * F == self.input_size, f'Incompatible LSTM inputs ({B}x{C}x{F} != {self.input_size})'
        x = torch.reshape(inputs, (T, N, B*C*F))
        # (T, N, hidden_size)
        x, _ = self.lstm(x)
        return x


class BandedConv2D(nn.Module):
    """A 2D temporal convolution block that respects the bands.

    Args:
        num_bands (int): Number of bands in use.
        channels_in (int): Number of input channels; for the first layer,
            should be the number of sensors on each band.
        channels_out (int): Number of filters.
        num_freq_bins (int): the number of frequency bins.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, num_bands: int, in_channels: int, out_channels: int, num_freq_bins: int, kernel_width: int) -> None:
        super().__init__()
        assert num_bands > 1, "Don't need banded convolution with 1 band."
        self.num_bands = num_bands
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_freq_bins = num_freq_bins
        self.kernel_width = kernel_width

        self.conv2ds = []
        self.relus = []
        self.layer_norms = []
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        for _ in range(self.num_bands):
            model = nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=(1, kernel_width),
                    )
            model.to(device)
            self.conv2ds.append(model)
            model = nn.ReLU()
            model.to(device)
            self.relus.append(model)
            model = nn.LayerNorm((self.out_channels, self.num_freq_bins))
            model.to(device)
            self.layer_norms.append(model)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, B, C, f = inputs.shape  # TNC

        stack = []
        for band in range(self.num_bands):
            # T, N, C, f
            x = inputs[:, :, band, :, :]
            # TNCf -> NCfT
            x = x.movedim(0, -1)
            self.conv2ds[band].to(x.device)
            x = self.conv2ds[band](x)
            x = self.relus[band](x)
            # NCfT -> TNCf
            x = x.movedim(-1, 0)
            x = self.layer_norms[band](x)
            stack.append(x)

        return torch.stack(stack, dim=2)