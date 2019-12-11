from config import ConfigArgs as args
import torch
import torch.nn as nn
from network import TextEncoder, ReferenceEncoder, StyleTokenLayer, AudioDecoder
import module as mm

class TPGST(nn.Module):
    """
    GST-Tacotron
   
    """
    def __init__(self):
        super(TPGST, self).__init__()
        self.name = 'TPGST'
        self.embed = nn.Embedding(len(args.vocab), args.Ce, padding_idx=0)
        self.encoder = TextEncoder(hidden_dims=args.Cx) # bidirectional
        self.GST = StyleTokenLayer(embed_size=args.Cx, n_units=args.Cx)
        self.tpnet = TPSENet(text_dims=args.Cx*2, style_dims=args.Cx)
        self.decoder = AudioDecoder(enc_dim=args.Cx*3, dec_dim=args.Cx)
    
    def forward(self, texts, prev_mels, refs=None, synth=False, ref_mode=True):
        """
        :param texts: (N, Tx) Tensor containing texts
        :param prev_mels: (N, Ty/r, n_mels*r) Tensor containing previous audio
        :param refs: (N, Ty, n_mels) Tensor containing reference audio
        :param synth: Boolean. whether it synthesizes.
        :param ref_mode: Boolean. whether it is reference mode

        Returns:
            :mels_hat: (N, Ty/r, n_mels*r) Tensor. mel spectrogram
            :mags_hat: (N, Ty, n_mags) Tensor. magnitude spectrogram
            :attentions: (N, Ty/r, Tx) Tensor. seq2seq attention
            :style_attentions: (N, n_tokens) Tensor. Style token layer attention
            :ff_hat: (N, Ty/r, 1) Tensor for binary final prediction
            :style_emb: (N, 1, E) Tensor. Style embedding
        """
        x = self.embed(texts)  # (N, Tx, Ce)
        text_emb, enc_hidden = self.encoder(x) # (N, Tx, Cx*2)
        tp_style_emb = self.tpnet(text_emb)
        if synth:
            style_emb, style_attentions = tp_style_emb, None
        else:
            style_emb, style_attentions = self.GST(refs, ref_mode=ref_mode) # (N, 1, E), (N, n_tokens)
        
        tiled_style_emb = style_emb.expand(-1, text_emb.size(1), -1) # (N, Tx, E)
        memory = torch.cat([text_emb, tiled_style_emb], dim=-1)  # (N, Tx, Cx*2+E)
        mels_hat, mags_hat, attentions, ff_hat = self.decoder(prev_mels, memory, synth=synth)
        return mels_hat, mags_hat, attentions, style_attentions, ff_hat, style_emb, tp_style_emb

class TPSENet(nn.Module):
    """
    Predict speakers from style embedding (N-way classifier)

    """
    def __init__(self, text_dims, style_dims):
        super(TPSENet, self).__init__()
        self.conv = nn.Sequential(
            mm.Conv1d(text_dims, style_dims, 3, activation_fn=torch.relu, bn=True, bias=False),
            # mm.Conv1d(style_dims, style_dims, 3, activation_fn=torch.relu, bn=True, bias=False)
        )
        self.gru = nn.GRU(style_dims, style_dims, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(style_dims*2, style_dims)
        # self.net = nn.Linear(args.Cx, args.n_speakers)
    
    def forward(self, text_embedding):
        """
        :param text_embedding: (N, Tx, E)

        Returns:
            :y_: (N, 1, n_speakers) Tensor.
        """
        te = text_embedding.transpose(1, 2) # (N, E, Tx)
        h = self.conv(te)
        h = h.transpose(1, 2) # (N, Tx, C)
        out, _ = self.gru(h)
        se = self.fc(out[:, -1:, :])
        se = torch.tanh(se)
        return se

