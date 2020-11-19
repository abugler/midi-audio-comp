import torch
import torch.nn as nn

class Matcher(nn.Module):
    """
    Have two BLSTM nn, and return whether the they are from the same source
    Parmeters are specifically chosen for a 8 second clip, 16k sr, and 512 window length
    input:
        spectrogram: (batch, n_freq, n_ts)
        midi: (batch, n_pitch, n_ts)
    output:
        decision: (batch, same)
    """
    def __init__(
        self,
        spectrogram_dim,
        midi_dim,
        spectrogram_hidden_dim,
        midi_hidden_dim,
        num_layers,
        dropout,
        spectrogram_kernels,
        midi_kernels,
        spectrogram_channels,
        midi_channels,
        device):
        super().__init__()
        self.__dict__.update(locals())

        # LSTM was stripped out
        # self.lstm_s = nn.LSTM(
        #     spectrogram_dim, spectrogram_hidden_dim, num_layers, dropout=dropout,
        #     bidirectional=True
        #     )
        # self.lstm_m = nn.LSTM(
        #     midi_dim, midi_hidden_dim, num_layers, dropout=dropout,
        #     bidirectional=True
        #     )

        self.spectrogram_conv = Matcher._make_conv_stack(
            spectrogram_channels, spectrogram_kernels
        )
        self.midi_conv = Matcher._make_conv_stack(
            midi_channels, midi_kernels
        )
        out_m_dim= Matcher._find_output_dim(
            midi_dim, # if inserting LSTM, replace with midi_hidden_dim * 2
            midi_channels,
            midi_kernels
        )
        out_s_dim = Matcher._find_output_dim(
            spectrogram_dim, # if inserting LSTM, replace with spectrogram_hidden_dim * 2
            spectrogram_channels,
            spectrogram_kernels
        )
        print(f"input to linear layer is size {out_m_dim + out_s_dim}")
        self.linear = nn.Linear(out_m_dim + out_s_dim, 1)
        self.sigmoid = nn.Sigmoid()
        super().to(device)


    @staticmethod
    def _make_conv_stack(channels, kernels):
        return nn.Sequential(*[
            nn.Conv2d(
                in_, out, k, stride=2
            ) for (in_, out), k
            in zip(Matcher._channel_pairs(channels), kernels)
        ])

    @staticmethod
    def _find_output_dim(inp_dim, channels, kernels):
        x = inp_dim
        y = 1001 # hard coded
        for k in kernels:
            if isinstance(k, int):
                kx, ky = k, k
            else:
                kx, ky = k[0], k[1]
            x = int((x - (kx - 1) - 1)/2 + 1)
            y = int((y - (ky - 1) - 1)/2 + 1)
        c = channels[-1]
        return x * y * c

    @staticmethod
    def _channel_pairs(channels):
        return [
            (1 if i == 0 else channels[i-1], channels[i])
            for i in range(len(channels))
        ]


    def get_rnn_states(self, batch_size):
        hs_0 = torch.zeros((self.num_layers * 2,
            batch_size, self.spectrogram_hidden_dim)).to(self.device)
        cs_0 = torch.zeros((self.num_layers * 2,
            batch_size, self.spectrogram_hidden_dim)).to(self.device)
        hm_0 = torch.zeros((self.num_layers * 2,
            batch_size, self.midi_hidden_dim)).to(self.device)
        cm_0 = torch.zeros((self.num_layers * 2,
            batch_size, self.midi_hidden_dim)).to(self.device)
        return hs_0, cs_0, hm_0, cm_0


    def forward(self, spectrogram, midi):
        # batch_size = spectrogram.size()[0]
        # spectrogram, midi = spectrogram.permute(2, 0, 1), midi.permute(2, 0, 1)
        # hs_0, cs_0, hm_0, cm_0 = self.get_rnn_states(batch_size)
        # s_out, _ = self.lstm_s(spectrogram, (hs_0, cs_0))
        # s_out = s_out.permute(1, 2, 0).unsqueeze(1)
        # m_out, _ = self.lstm_m(midi, (hm_0, cm_0))
        # m_out = m_out.permute(1, 2, 0).unsqueeze(1)

        # Comment out these two lines, and uncomment all lines before it for LSTM
        s_out = spectrogram.unsqueeze(dim=1)
        m_out = midi.unsqueeze(dim=1)


        s_out = torch.flatten(self.spectrogram_conv(s_out), start_dim=1)
        m_out = torch.flatten(self.midi_conv(m_out), start_dim=1)
        out = self.linear(torch.cat((s_out, m_out), dim=-1))
        return self.sigmoid(out).squeeze(dim=-1)
