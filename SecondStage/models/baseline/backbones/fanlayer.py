"""

"""


class EnhancedFANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, p_ratio=0.15, activation='gelu',
                 use_p_bias=True, use_residual=True, wavelet_type='learnable'):
        super(EnhancedFANLayer, self).__init__()

        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        self.use_residual = use_residual
        self.wavelet_type = wavelet_type

        #
        p_output_dim = int(output_dim * self.p_ratio)
        self.g_output_dim = output_dim - 2 * p_output_dim  #

        #
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)

        #
        self.input_linear_g = nn.Linear(input_dim, self.g_output_dim)

        #
        nn.init.xavier_normal_(self.input_linear_p.weight)
        if use_p_bias:
            nn.init.constant_(self.input_linear_p.bias, 0.0)
        nn.init.kaiming_normal_(self.input_linear_g.weight, nonlinearity='relu')

        # ）
        if wavelet_type == 'learnable':
            self.freq = nn.Parameter(torch.randn(p_output_dim) * 5 + 5  #
            self.scale = nn.Parameter(torch.ones(p_output_dim))  #
            elif wavelet_type == 'morlet':
            self.register_buffer('freq', torch.ones(p_output_dim) * 5)  #
            self.register_buffer('scale', torch.ones(p_output_dim))

            #
            if activation == 'gelu':
                self.activation = F.gelu
            elif activation == 'silu':
                self.activation = F.silu
            else:
                self.activation = getattr(F, activation) if isinstance(activation, str) else activation

            #
            if use_residual and input_dim != output_dim:
                self.res_linear = nn.Linear(input_dim, output_dim)
            else:
                self.res_linear = None

    def wavelet_transform(self, x):

        if self.wavelet_type == 'learnable':
            # ）
            return torch.cos(self.freq * x) * torch.exp(-x ** 2 / (2 * self.scale ** 2))
        elif self.wavelet_type == 'mexican':
            #
            return (1 - (x / self.scale) ** 2) * torch.exp(-x ** 2 / (2 * self.scale ** 2))
        else:
            #
            return torch.cos(5 * x) * torch.exp(-x ** 2 / 2)

    def forward(self, src):
        residual = src
        if self.use_residual and self.res_linear is not None:
            residual = self.res_linear(residual)

        #
        g = self.activation(self.input_linear_g(src))

        #
        p = self.input_linear_p(src)
        wavelet_cos = self.wavelet_transform(p)
        wavelet_sin = self.wavelet_transform(-p)  #

        #
        output = torch.cat([wavelet_cos, wavelet_sin, g], dim=-1)

        #
        if self.use_residual:
            output += residual

        return output


class MultiLevelWaveletFAN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, levels=3,
                 p_ratio=0.2, activation='silu'):
        super().__init__()
        self.wavelet_layers = nn.ModuleList([
            EnhancedFANLayer(
                input_dim=input_dim if i == 0 else hidden_dim,
                output_dim=hidden_dim,
                p_ratio=p_ratio,
                activation=activation,
                wavelet_type='learnable',
                use_residual=True
            ) for i in range(levels)
        ])
        self.final_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.wavelet_layers:
            x = layer(x)
        return self.final_proj(x)
        """
"""
