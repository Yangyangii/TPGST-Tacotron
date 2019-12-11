from config import ConfigArgs as args
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils import weight_norm as norm
import numpy as np
import module as mm

class PreNet(nn.Module):
    """

    :param input_dim: Scalar.
    :param hidden_dim: Scalar.

    """
    def __init__(self, input_dim, hidden_dim):
        super(PreNet, self).__init__()
        self.prenet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        """
        :param x: (N, Tx, Ce) Tensor.

        Returns:
            y_: (N, Tx, Cx) Tensor.

        """
        y_ = self.prenet(x)
        return y_


class CBHG(nn.Module):
    """
    CBHG module (Convolution bank + Highwaynet + GRU)

    :param input_dim: Scalar.
    :param hidden_dim: Scalar.
    :param K: Scalar. K sets of 1-D conv filters
    :param n_highway: Scalar. number of highway layers
    :param bidirectional: Boolean. whether it is bidirectional

    """

    def __init__(self, input_dim, hidden_dim, K=16, n_highway=4, bidirectional=True):
        super(CBHG, self).__init__()
        self.K = K
        self.conv_bank = mm.Conv1dBank(input_dim, hidden_dim, K=self.K, activation_fn=torch.relu)
        self.max_pool = nn.MaxPool1d(2, stride=1, padding=1)
        self.projection = nn.Sequential(
            mm.Conv1d(self.K*hidden_dim, hidden_dim, 3, activation_fn=torch.relu, bias=False, bn=True),
            mm.Conv1d(hidden_dim, input_dim, 3, bias=False, bn=True),
        )
        self.highway = nn.ModuleList(
            [mm.Highway(input_dim) for _ in range(n_highway)]
        )
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirectional) # if batch_first is True, (Batch, Sequence, Feature)

    def forward(self, x, prev=None):
        """
        :param x: (N, T, input_dim) Tensor.
        :param prev: Tensor. for gru

        Returns:
            :y_: (N, T, 2*hidden_dim) Tensor.
            :hidden: Tensor. hidden state

        """
        y_ = x.transpose(1, 2) # (N, input_dim, Tx)
        y_ = self.conv_bank(y_) # (N, K*hidden_dim, Tx)
        y_ = self.max_pool(y_)[:, :, :-1] # pooling over time
        y_ = self.projection(y_) # (N, input_dim, Tx)
        y_ = y_.transpose(1, 2) # (N, Tx, input_dim)
        # Residual connection
        y_ = y_ + x  # (N, Tx, input_dim)
        for idx in range(len(self.highway)):
            y_ = self.highway[idx](y_)  # (N, Tx, input_dim)
        y_, hidden = self.gru(y_, prev)  # (N, Tx, hidden_dim)
        return y_, hidden


class TextEncoder(nn.Module):
    """
    Text Encoder
    Prenet + CBHG

    """
    def __init__(self, hidden_dims):
        super(TextEncoder, self).__init__()
        self.prenet = PreNet(args.Ce, hidden_dims)
        self.cbhg = CBHG(input_dim=hidden_dims, hidden_dim=hidden_dims, K=16, n_highway=4, bidirectional=True)

    def forward(self, x):
        """
        :param x: (N, Tx, Ce) Tensor. Character embedding

        Returns:
            :y_: (N, Tx, 2*Cx) Text Embedding
            :hidden: Tensor.
        
        """
        y_ = self.prenet(x)  # (N, Tx, Cx)
        y_, hidden = self.cbhg(y_)  # (N, Cx*2, Tx)
        return y_, hidden


class ReferenceEncoder(nn.Module):
    """
    Reference Encoder.
    6 convs + GRU + FC

    :param in_channels: Scalar.
    :param embed_size: Scalar.
    :param activation_fn: activation function

    """
    def __init__(self, in_channels=1, embed_size=128, activation_fn=None):
        super(ReferenceEncoder, self).__init__()
        self.embed_size = embed_size
        channels = [in_channels, 32, 32, 64, 64, 128, embed_size]
        self.convs = nn.ModuleList([
            mm.Conv2d(channels[c], channels[c+1], 3, stride=2, bn=True, bias=False, activation_fn=torch.relu)
            for c in range(len(channels)-1)
        ]) # (N, Ty/r, 128)
        self.gru = nn.GRU(self.embed_size*2, self.embed_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size),
        )
        self.activation_fn = activation_fn

    def forward(self, x, hidden=None):
        """
        :param x: (N, 1, Ty, n_mels) Tensor. Mel Spectrogram
        :param hidden: Tensor. initial hidden state for gru

        Returns:
            y_: (N, 1, E) Reference Embedding

        """
        y_ = x
        for i in range(len(self.convs)):
            y_ = self.convs[i](y_)
        # (N, C, Ty//64, n_mels//64)
        y_ = y_.transpose(1, 2) # (N, Ty//64, C, n_mels//64)
        shape = y_.shape
        y_ = y_.contiguous().view(shape[0], -1, shape[2]*shape[3]) # (N, Ty//64, C*n_mels//64)
        y_, out = self.gru(y_, hidden) # (N, Ty//64, E)
        # y_ = y_[:, -1, :] # (N, E)
        y_ = out.squeeze(0) # same as states[:, -1, :]
        y_ = self.fc(y_) # (N, E)
        y_ = self.activation_fn(y_) if self.activation_fn is not None else y_
        return y_.unsqueeze(1)

class StyleTokenLayer(nn.Module):
    """
    Style Token Layer
    Reference Encoder + Multi-head Attention, token embeddings

    :param embed_size: Scalar.
    :param n_units: Scalar. for multihead attention ***

    """
    def __init__(self, embed_size=128, n_units=128):
        super(StyleTokenLayer, self).__init__()
        self.token_embedding = nn.Parameter(torch.zeros([args.n_tokens, embed_size])) # (n_tokens, E)
        self.ref_encoder = ReferenceEncoder(in_channels=1, embed_size=embed_size, activation_fn=torch.tanh)
        self.att = MultiHeadAttention(n_units, embed_size)

        init.normal_(self.token_embedding, mean=0., std=0.5)
        # init.orthogonal_(self.token_embedding)
 
    def forward(self, ref, ref_mode=True):
        """
        :param ref: (N, Ty, n_mels) Tensor containing reference audio or (N, n_tokens) if not ref_mode
        :param ref_mode: Boolean. whether it is reference mode

        Returns:
            :y_: (N, 1, E) Style embedding
            :A: (N, n_tokens) Tensor. Combination weight.

        """
        token_embedding = self.token_embedding.unsqueeze(0).expand(ref.size(0), -1, -1) # (N, n_tokens, E)
        if ref_mode:
            ref = self.ref_encoder(ref) # (N, 1, E)
            A = torch.softmax(self.att(ref, token_embedding), dim=-1) # (N, n_tokens)
            # A = torch.softmax(self.att(ref, token_embedding)) # (N, n_tokens)
        else:
            A = torch.softmax(ref, dim=-1)
        y_ = torch.sum(A.unsqueeze(-1) * token_embedding, dim=1, keepdim=True) # (N, 1, E)
        y_ = torch.tanh(y_)
        return y_, A

class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention

    :param n_units: Scalars.
    :param embed_size : Scalars.

    """
    def __init__(self, n_units=128, embed_size=128):
        super(MultiHeadAttention, self).__init__()
        self.split_size = n_units // args.n_heads
        self.conv_Q = mm.Conv1d(embed_size, n_units, 1)
        self.conv_K = mm.Conv1d(embed_size, n_units, 1)
        self.fc_Q = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.Tanh(),
        )
        self.fc_K = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.Tanh(),
        )
        self.fc_V = nn.Sequential(
            nn.Linear(embed_size, self.split_size),
            nn.Tanh(),
        )
        self.fc_A = nn.Sequential(
            nn.Linear(n_units, args.n_tokens),
            nn.Tanh(),
        )
        

    def forward(self, ref_embedding, token_embedding):
        """
        :param ref_embedding: (N, 1, E) Reference embedding
        :param token_embedding: (N, n_tokens, embed_size) Token Embedding

        Returns:
            y_: (N, n_tokens) Tensor. Style attention weight

        """
        Q = self.fc_Q(self.conv_Q(ref_embedding.transpose(1,2)).transpose(1,2))  # (N, 1, n_units)
        K = self.fc_K(self.conv_K(token_embedding.transpose(1,2)).transpose(1,2))  # (N, n_tokens, n_units)
        V = self.fc_V(token_embedding)  # (N, n_tokens, n_units)
        Q = torch.stack(Q.split(self.split_size, dim=-1), dim=0) # (n_heads, N, 1, n_units//n_heads)
        K = torch.stack(K.split(self.split_size, dim=-1), dim=0) # (n_heads, N, n_tokens, n_units//n_heads)
        V = torch.stack(V.split(self.split_size, dim=-1), dim=0) # (n_heads, N, n_tokens, n_units//n_heads)
        inner_A = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / self.split_size**0.5,
            dim=-1
        ) # (n_heads, N, 1, n_tokens)
        y_ = torch.matmul(inner_A, V)  # (n_heads, N, 1, n_units//n_heads)
        y_ = torch.cat(y_.split(1, dim=0), dim=-1).squeeze() # (N, n_units)
        y_ = self.fc_A(y_) # (N, n_tokens)
        return y_


class AudioDecoder(nn.Module):
    """
    Audio Decoder
    prenet + attention RNN + 2 RNN + FC + CBHG

    :param enc_dim: Scalar. for encoder output
    :param dec_dim: Scalar. for decoder input

    """
    def __init__(self, enc_dim, dec_dim):
        super(AudioDecoder, self).__init__()
        self.prenet = PreNet(args.n_mels*args.r, dec_dim)
        self.attention_rnn = mm.AttentionRNN(enc_dim=enc_dim, dec_dim=dec_dim)
        self.proj_att = nn.Linear(enc_dim+dec_dim, dec_dim)
        self.decoder_rnn = nn.ModuleList([
            nn.GRU(dec_dim, dec_dim, num_layers=1, batch_first=True, bidirectional=False)
            for _ in range(2)
        ])
        self.final_frame = nn.Sequential(
            nn.Linear(dec_dim, 1),
            nn.Sigmoid(),
        )
        self.proj_mel = nn.Linear(dec_dim, args.n_mels*args.r)
        self.cbhg = CBHG(input_dim=args.n_mels, hidden_dim=dec_dim//2, K=8, n_highway=4, bidirectional=True)
        self.proj_mag = nn.Linear(dec_dim, args.n_mels)

    def forward(self, decoder_inputs, encoder_outputs, prev_hidden=None, synth=False):
        """
        :param decoder_inputs: (N, Ty/r, n_mels*r) Tensor. Decoder inputs (previous decoder outputs)
        :param encoder_outputs: (N, Tx, Cx) Tensor. Encoder output *** general??
        :param prev_hidden: Tensor. hidden state for gru when synth is true
        :param synth: Boolean. whether it synthesizes

        Returns:
            :mels_hat: (N, Ty/r, n_mels*r) Mel spectrogram
            :mags_hat: (N, Ty, n_mags) Magnitude spectrogram
            :A: (N, Ty/r, Tx) Tensor. Attention weights
            :ff_hat: (N, Ty/r, 1) Tensor. for binary final frame prediction
            
        """
        if not synth: # Train mode & Eval mode
            y_ = self.prenet(decoder_inputs) # (N, Ty/r, Cx)
            # Attention RNN
            y_, A, hidden = self.attention_rnn(encoder_outputs, y_) # y_: (N, Ty/r, Cx), A: (N, Ty/r, Tx)
            # (N, Ty/r, Tx) . (N, Tx, Cx)
            c = torch.matmul(A, encoder_outputs) # (N, Ty/r, Cx)
            y_ = self.proj_att(torch.cat([c, y_], dim=-1)) # (N, Ty/r, Cx)

            # Decoder RNN
            for idx in range(len(self.decoder_rnn)):
                y_f, _ = self.decoder_rnn[idx](y_)  # (N, Ty/r, Cx)
                y_ = y_ + y_f

            # binary final frame prediction
            ff_hat = torch.clamp(self.final_frame(y_)+1e-10, 1e-10, 1) # (N, Ty/r, 1)

            # Mel-spectrogram
            mels_hat = self.proj_mel(y_) # (N, Ty/r, n_mels*r)

            # Decoder CBHG
            y_ = mels_hat.view(mels_hat.size(0), -1, args.n_mels) # (N, Ty, n_mels)
            y_, _ = self.cbhg(y_)  # (N, Ty, Cx*2)
            mags_hat = self.proj_mag(y_) # (N, Ty, n_mags)
            return mels_hat, mags_hat, A, ff_hat
        else:
            # decoder_inputs: GO frame (N, 1, n_mels*r)
            att_hidden = None
            dec_hidden = [None, None]

            mels_hat = []
            mags_hat = []
            attention = []
            for idx in range(args.max_Ty):
                y_ = self.prenet(decoder_inputs)  # (N, 1, Cx)
                # Attention RNN
                y_, A, att_hidden = self.attention_rnn(encoder_outputs, y_, prev_hidden=att_hidden)
                attention.append(A)
                # Encoder outputs: (N, Tx, Cx)
                # A: (N, )
                c = torch.matmul(A, encoder_outputs)  # (N, Ty/r, Cx)
                y_ = self.proj_att(torch.cat([c, y_], dim=-1))  # (N, Ty/r, Cx)

                # Decoder RNN
                for j in range(len(self.decoder_rnn)):
                    y_f, dec_hidden[j] = self.decoder_rnn[j](y_, dec_hidden[j])  # (N, 1, Cx)
                    y_ = y_ + y_f  # (N, 1, Cx)

                # binary final frame prediction
                ff_hat = self.final_frame(y_) # (N, Ty/r, 1)

                # Mel-spectrogram
                mel_hat = self.proj_mel(y_)  # (N, 1, n_mels*r)
                decoder_inputs = mel_hat[:, :, -args.n_mels*args.r:] # last frame
                mels_hat.append(mel_hat)
                
                if (ff_hat[:, -1] > 0.5).sum() == len(ff_hat):
                    break
            
            mels_hat = torch.cat(mels_hat, dim=1)
            attention = torch.cat(attention, dim=1)

            # Decoder CBHG
            y_ = mels_hat.view(mels_hat.size(0), -1, args.n_mels) # (N, Ty, n_mels)
            y_, _ = self.cbhg(y_)
            mags_hat = self.proj_mag(y_)
            
            return mels_hat, mags_hat, attention, ff_hat

