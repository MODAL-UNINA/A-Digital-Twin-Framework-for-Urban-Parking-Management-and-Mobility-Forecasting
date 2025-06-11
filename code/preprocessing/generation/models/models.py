from typing import TypedDict

import torch
from torch import nn


class ModelArgs(TypedDict):
    input_dim: int
    cond_dim: int
    hidden_dim: int
    latent_dim: int
    kernel_size: int
    padding: int
    horizon: int
    grid_size: int
    use_proximity: bool


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        activation: str,
        frame_size: list[int],
    ) -> None:
        super(ConvLSTMCell, self).__init__()  # type: ignore
        self.init_hidden_dim = out_channels

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        with torch.no_grad():
            assert self.conv.bias is not None, "Conv2d bias should not be None"
            nn.init.constant_(self.conv.bias[out_channels : 2 * out_channels], 1.0)

        self.norm_i = nn.LayerNorm([out_channels, *frame_size])
        self.norm_f = nn.LayerNorm([out_channels, *frame_size])
        self.norm_o = nn.LayerNorm([out_channels, *frame_size])

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.kaiming_normal_(self.W_ci)
        nn.init.kaiming_normal_(self.W_co)
        nn.init.kaiming_normal_(self.W_cf)

    def forward(
        self, X: torch.Tensor, H_prev: torch.Tensor, C_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C


class CrossAttention2D(nn.Module):
    def __init__(
        self, query_dim: int, key_dim: int, value_dim: int, hidden_dim: int
    ) -> None:
        super(CrossAttention2D, self).__init__()  # type: ignore
        self.query_conv = nn.Conv2d(query_dim, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(key_dim, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(value_dim, hidden_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv2d(hidden_dim, query_dim, kernel_size=1)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query: [B, query_dim, H, W]
        key: [B, key_dim, H, W]
        value: [B, value_dim, H, W]
        """
        Q = self.query_conv(query)
        K = self.key_conv(key)
        V = self.value_conv(value)

        B, C, H, W = Q.shape
        Q_flat = Q.view(B, C, H * W)
        K_flat = K.view(B, C, H * W)
        V_flat = V.view(B, C, H * W)

        attention_scores = torch.bmm(Q_flat.transpose(1, 2), K_flat)
        attention_weights = self.softmax(attention_scores)

        context = torch.bmm(V_flat, attention_weights.transpose(1, 2))
        context = context.view(B, C, H, W)

        output = self.output_conv(context)

        return output, attention_weights


class Encoder(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super(Encoder, self).__init__()  # type: ignore

        self.input_dim = model_args["cond_dim"]
        self.hidden_dim = model_args["hidden_dim"]
        self.latent_dim = model_args["latent_dim"]
        self.kernel = model_args["kernel_size"]
        self.padding = model_args["padding"]
        self.horizon = model_args["horizon"]
        self.grid_size = model_args["grid_size"]
        self.last_hidden_dim = self.hidden_dim * 8
        self.kernel_size = (1, self.kernel, self.kernel)
        self.padding_size = (0, self.padding, self.padding)
        self.final_horizon = self.horizon // 8
        self.height = self.grid_size // 16
        self.width = self.grid_size // 16

        # Convolutional layers with pooling
        self.conv1d_1 = nn.Conv3d(
            self.input_dim,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn1 = nn.BatchNorm3d(self.hidden_dim)
        self.relu1 = nn.ReLU()

        self.conv1d_2 = nn.Conv3d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn2 = nn.BatchNorm3d(self.hidden_dim)
        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True
        )

        self.conv1d_3 = nn.Conv3d(
            self.hidden_dim,
            self.hidden_dim * 2,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn3 = nn.BatchNorm3d(self.hidden_dim * 2)
        self.relu3 = nn.ReLU()

        self.conv1d_4 = nn.Conv3d(
            self.hidden_dim * 2,
            self.hidden_dim * 2,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn4 = nn.BatchNorm3d(self.hidden_dim * 2)
        self.relu4 = nn.ReLU()

        self.pool4 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True
        )

        self.conv1d_5 = nn.Conv3d(
            self.hidden_dim * 2,
            self.hidden_dim * 4,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn5 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu5 = nn.ReLU()

        self.conv1d_6 = nn.Conv3d(
            self.hidden_dim * 4,
            self.hidden_dim * 4,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn6 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu6 = nn.ReLU()

        self.conv1d_7 = nn.Conv3d(
            self.hidden_dim * 4,
            self.hidden_dim * 4,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn7 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu7 = nn.ReLU()

        self.pool7 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True
        )

        self.conv1d_8 = nn.Conv3d(
            self.hidden_dim * 4,
            self.last_hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn8 = nn.BatchNorm3d(self.last_hidden_dim)
        self.relu8 = nn.ReLU()

        self.conv1d_9 = nn.Conv3d(
            self.last_hidden_dim,
            self.last_hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn9 = nn.BatchNorm3d(self.last_hidden_dim)
        self.relu9 = nn.ReLU()

        self.conv1d_10 = nn.Conv3d(
            self.last_hidden_dim,
            self.last_hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn10 = nn.BatchNorm3d(self.last_hidden_dim)
        self.relu10 = nn.ReLU()

        self.pool10 = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True
        )

        # Convolutional LSTM
        self.dim_init = [
            self.last_hidden_dim,
            self.last_hidden_dim * 2,
            self.last_hidden_dim * 2,
        ]
        self.dim_final = [
            self.last_hidden_dim * 2,
            self.last_hidden_dim * 2,
            self.last_hidden_dim * 2,
        ]
        self.lstm_layers = nn.ModuleList(
            [
                ConvLSTMCell(
                    in_channels=self.dim_init[i],
                    out_channels=self.dim_final[i],
                    kernel_size=self.kernel,
                    padding=self.padding,
                    activation="tanh",
                    frame_size=[self.height, self.width],
                )
                for i in range(len(self.dim_init))
            ]
        )

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(
            self.last_hidden_dim * 2 * self.height * self.width, self.latent_dim
        )
        self.fc_logvar = nn.Linear(
            self.last_hidden_dim * 2 * self.height * self.width, self.latent_dim
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor
    ]:
        pool_idx: list[torch.Tensor] = []
        seq_len = x.shape[2]

        # Convolutional block 1
        h = self.relu1(self.bn1(self.conv1d_1(x)))
        h = self.relu2(self.bn2(self.conv1d_2(h)))
        h, pool_idx1 = self.pool2(h)
        pool_idx.append(pool_idx1)

        # Convolutional block 2
        h = self.relu3(self.bn3(self.conv1d_3(h)))
        h = self.relu4(self.bn4(self.conv1d_4(h)))
        h, pool_idx2 = self.pool4(h)
        pool_idx.append(pool_idx2)

        # Convolutional block 3
        h = self.relu5(self.bn5(self.conv1d_5(h)))
        h = self.relu6(self.bn6(self.conv1d_6(h)))
        h = self.relu7(self.bn7(self.conv1d_7(h)))
        h, pool_idx3 = self.pool7(h)
        pool_idx.append(pool_idx3)

        # Convolutional block 4
        h = self.relu8(self.bn8(self.conv1d_8(h)))
        h = self.relu9(self.bn9(self.conv1d_9(h)))
        h = self.relu10(self.bn10(self.conv1d_10(h)))
        h, pool_idx4 = self.pool10(h)
        pool_idx.append(pool_idx4)
        y = h

        # Convolutional LSTM
        for i, lstm_layer in enumerate(self.lstm_layers):
            output_matrix = torch.zeros(
                h.shape[0], self.dim_final[i], seq_len, self.height, self.width
            ).to(h.device)

            # Initialize hidden state
            tH = torch.zeros(h.shape[0], self.dim_final[i], self.height, self.width).to(
                h.device
            )

            # Initialize cell state
            tC = torch.zeros(h.shape[0], self.dim_final[i], self.height, self.width).to(
                h.device
            )

            for t in range(seq_len):
                tH, tC = lstm_layer(h[:, :, t], tH, tC)
                output_matrix[:, :, t] = tH

            h = output_matrix

        # Last map
        h = h[:, :, -1]

        flattened = self.flatten(h)
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, pool_idx, y


class Generator(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super(Generator, self).__init__()  # type: ignore

        self.input_dim = model_args["input_dim"]
        self.hidden_dim = model_args["hidden_dim"]
        self.latent_dim = model_args["latent_dim"]
        self.kernel = model_args["kernel_size"]
        self.padding = model_args["padding"]
        self.horizon = model_args["horizon"]
        self.grid_size = model_args["grid_size"]
        self.last_hidden_dim = self.hidden_dim * 8
        self.kernel_size = (1, self.kernel, self.kernel)
        self.padding_size = (0, self.padding, self.padding)
        self.final_horizon = self.horizon // 8
        self.height = self.grid_size // 16
        self.width = self.grid_size // 16

        # From latent space to map
        self.initial_fc = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(
            self.latent_dim, self.last_hidden_dim * 2 * self.height * self.width
        )

        self.in_sizes = [
            self.last_hidden_dim * 2 + self.last_hidden_dim,
            self.last_hidden_dim * 2,
            self.last_hidden_dim * 2,
        ]
        self.out_sizes = [
            self.last_hidden_dim * 2,
            self.last_hidden_dim * 2,
            self.last_hidden_dim,
        ]

        # Convolutional LSTM
        self.lstm_layers = nn.ModuleList(
            [
                ConvLSTMCell(
                    in_channels=self.in_sizes[i],
                    out_channels=self.out_sizes[i],
                    kernel_size=self.kernel,
                    padding=self.padding,
                    activation="tanh",
                    frame_size=[self.height, self.width],
                )
                for i in range(len(self.in_sizes))
            ]
        )

        # Cross-Attention
        self.cross_attention = nn.ModuleList(
            [
                CrossAttention2D(
                    query_dim=self.out_sizes[i],
                    key_dim=self.last_hidden_dim,
                    value_dim=self.last_hidden_dim,
                    hidden_dim=self.out_sizes[i],
                )
                for i in range(len(self.in_sizes))
            ]
        )

        # Convolutional layers
        self.conv1 = nn.Conv3d(
            self.last_hidden_dim,
            self.last_hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn1 = nn.BatchNorm3d(self.last_hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(
            self.last_hidden_dim,
            self.last_hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn2 = nn.BatchNorm3d(self.last_hidden_dim)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(
            self.last_hidden_dim,
            self.hidden_dim * 4,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn3 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(
            self.hidden_dim * 4,
            self.hidden_dim * 4,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn4 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv3d(
            self.hidden_dim * 4,
            self.hidden_dim * 4,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn5 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv3d(
            self.hidden_dim * 4,
            self.hidden_dim * 2,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn6 = nn.BatchNorm3d(self.hidden_dim * 2)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv3d(
            self.hidden_dim * 2,
            self.hidden_dim * 2,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn7 = nn.BatchNorm3d(self.hidden_dim * 2)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv3d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn8 = nn.BatchNorm3d(self.hidden_dim)
        self.relu8 = nn.ReLU()

        self.conv9 = nn.Conv3d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn9 = nn.BatchNorm3d(self.hidden_dim)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv3d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn10 = nn.BatchNorm3d(self.hidden_dim)
        self.relu10 = nn.ReLU()

        self.final_conv = nn.Conv3d(
            self.hidden_dim,
            self.input_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.unpool = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.final_activation = nn.Sigmoid()

    def forward(
        self, z: torch.Tensor, last_map: torch.Tensor, indices: list[int]
    ) -> torch.Tensor:
        seq_len = last_map.shape[2]

        # FC layer
        z_expanded = self.fc(self.dropout(self.initial_fc(z))).view(
            z.shape[0], -1, self.height, self.width
        )
        x = z_expanded.unsqueeze(2).repeat(1, 1, seq_len, 1, 1)
        x = torch.cat([last_map, x], dim=1)

        # Convolutional LSTM
        output_state: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            # Initialize output
            output = torch.zeros(
                x.shape[0], self.out_sizes[i], seq_len, self.height, self.width
            ).to(x.device)

            # Initialize Hidden State
            tH = torch.zeros(x.shape[0], self.out_sizes[i], self.height, self.width).to(
                x.device
            )

            # Initialize Cell Input
            tC = torch.zeros(x.shape[0], self.out_sizes[i], self.height, self.width).to(
                x.device
            )

            for t in range(seq_len):
                tH, tC = lstm_layer(x[:, :, t], tH, tC)
                # Cross-Attention
                last_map_t = last_map[:, :, t, :, :]
                H_att, _ = self.cross_attention[i](tH, last_map_t, last_map_t)

                # Residual connection
                tH = tH + H_att
                output[:, :, t] = tH

            output_state.append((tH, tC))
            x = output

        # Convolutional block 1
        x = self.unpool(x, indices=indices[-1])
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        # Convolutional block 2
        x = self.unpool(x, indices=indices[-2], output_size=(self.horizon, 25, 25))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))

        # Convolutional block 3
        x = self.unpool(x, indices=indices[-3])
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.relu8(self.bn8(self.conv8(x)))

        # Convolutional block 4
        x = self.unpool(x, indices=indices[-4])
        x = self.relu9(self.bn9(self.conv9(x)))
        x = self.relu10(self.bn10(self.conv10(x)))

        h = self.final_conv(x)

        h = self.final_activation(h)

        return h


class Critic(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        super(Critic, self).__init__()  # type: ignore

        self.input_dim = model_args["input_dim"]
        self.cond_input = model_args["cond_dim"]
        self.hidden_dim = model_args["hidden_dim"]
        self.latent_dim = model_args["latent_dim"]
        self.kernel = model_args["kernel_size"]
        self.padding = model_args["padding"]
        self.horizon = model_args["horizon"]
        self.grid_size = model_args["grid_size"]
        self.use_proximity = model_args["use_proximity"]
        self.kernel_size = (self.kernel, self.kernel, self.kernel)
        self.padding_size = (self.padding, self.padding, self.padding)
        self.final_horizon = self.horizon // 8
        self.height = self.grid_size // 16
        self.width = self.grid_size // 16

        # Convolutional layers
        self.conv1d_1 = nn.Conv3d(
            self.input_dim,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn1 = nn.BatchNorm3d(self.hidden_dim)
        self.relu1 = nn.ReLU()

        self.conv1d_2 = nn.Conv3d(
            self.hidden_dim, self.hidden_dim, kernel_size=2, stride=2, padding=0
        )
        self.bn2 = nn.BatchNorm3d(self.hidden_dim)
        self.relu2 = nn.ReLU()

        self.conv1d_3 = nn.Conv3d(
            self.hidden_dim,
            self.hidden_dim * 2,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn3 = nn.BatchNorm3d(self.hidden_dim * 2)
        self.relu3 = nn.ReLU()

        self.conv1d_4 = nn.Conv3d(
            self.hidden_dim * 2, self.hidden_dim * 2, kernel_size=2, stride=2, padding=0
        )
        self.bn4 = nn.BatchNorm3d(self.hidden_dim * 2)
        self.relu4 = nn.ReLU()

        self.conv1d_5 = nn.Conv3d(
            self.hidden_dim * 2,
            self.hidden_dim * 4,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn5 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu5 = nn.ReLU()

        self.conv1d_6 = nn.Conv3d(
            self.hidden_dim * 4,
            self.hidden_dim * 4,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn6 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu6 = nn.ReLU()

        self.conv1d_7 = nn.Conv3d(
            self.hidden_dim * 4, self.hidden_dim * 4, kernel_size=2, stride=2, padding=0
        )
        self.bn7 = nn.BatchNorm3d(self.hidden_dim * 4)
        self.relu7 = nn.ReLU()

        self.conv1d_8 = nn.Conv3d(
            self.hidden_dim * 4,
            self.hidden_dim * 8,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn8 = nn.BatchNorm3d(self.hidden_dim * 8)
        self.relu8 = nn.ReLU()

        self.conv1d_9 = nn.Conv3d(
            self.hidden_dim * 8,
            self.hidden_dim * 8,
            kernel_size=self.kernel_size,
            stride=(1, 1, 1),
            padding=self.padding_size,
        )
        self.bn9 = nn.BatchNorm3d(self.hidden_dim * 8)
        self.relu9 = nn.ReLU()

        self.conv1d_10 = nn.Conv3d(
            self.hidden_dim * 8,
            self.hidden_dim * 8,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
        )

        self.bn10 = nn.BatchNorm3d(self.hidden_dim * 8)
        self.relu10 = nn.ReLU()

        if self.use_proximity:
            # Proximity layers
            self.proximity = nn.Conv3d(
                self.cond_input,
                self.hidden_dim,
                kernel_size=(1, 7, 7),
                stride=(1, 1, 1),
                padding=(0, 3, 3),
            )
            self.proximity_bn = nn.BatchNorm3d(self.hidden_dim)
            self.proximity_relu = nn.ReLU()
            self.proximity_fc = nn.Linear(
                self.hidden_dim * self.grid_size * self.grid_size * self.horizon,
                self.hidden_dim,
            )
            self.dropout_proximity = nn.Dropout(0.2)
            self.proximity_fc2 = nn.Linear(self.hidden_dim, 1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.hidden_dim * 8 * self.height * self.width * 5, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Convolutional block 1
        h = self.relu1(self.bn1(self.conv1d_1(x)))
        h = self.relu2(self.bn2(self.conv1d_2(h)))

        # Convolutional block 2
        h = self.relu3(self.bn3(self.conv1d_3(h)))
        h = self.relu4(self.bn4(self.conv1d_4(h)))

        # Convolutional block 3
        h = self.relu5(self.bn5(self.conv1d_5(h)))
        h = self.relu6(self.bn6(self.conv1d_6(h)))
        h = self.relu7(self.bn7(self.conv1d_7(h)))

        # Convolutional block 4
        h = self.relu8(self.bn8(self.conv1d_8(h)))
        h = self.relu9(self.bn9(self.conv1d_9(h)))
        h = self.relu10(self.bn10(self.conv1d_10(h)))

        z = self.fc(self.flatten(h))

        z = z.squeeze()
        if self.use_proximity:
            # Proximity
            proximity_features = self.proximity_relu(
                self.proximity_bn(self.proximity(mask))
            )
            proximity_features = self.flatten(proximity_features)
            proximity_features = self.dropout_proximity(
                self.proximity_fc(proximity_features)
            )
            proximity_features = self.proximity_fc2(proximity_features)

            z = z + proximity_features.squeeze()

        return z
