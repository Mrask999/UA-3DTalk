import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.hexplane import HexPlaneField
from scene.fusion_net import fusion_net
from scene.fusion_net_mouth import fusion_net_mouth
from scene.emotion_encoder import EmotionEncoder

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1)  # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size / 2)
        x = x[:, :, 8 - half_w:8 + half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class MotionNetwork(nn.Module):
    def __init__(self,audio_dim=32,):
        super(MotionNetwork, self).__init__()


        self.audio_in_dim = 29
        self.bound = 0.15
        self.exp_eye = True


        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        planeconfig = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 3,
            'output_coordinate_dim': 16,
            'resolution': [64, 64, 64]
        }
        multires = [1, 2, 4, 8]
        self.xyz_encoder =HexPlaneField(self.bound,planeconfig,multires)
        self.in_dim = planeconfig['output_coordinate_dim'] * len(multires)   # 4* 16 = 64
        self.num_layers = 3
        self.hidden_dim = 64
        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)
        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)
        emo_planeconfig = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 3,
            'output_coordinate_dim': 64,
            'resolution': [64, 64, 64]
        }
        emo_multires = [1, 2, 4]
        self.emotion_encoder = EmotionEncoder(self.bound, emo_planeconfig, emo_multires)
        self.out_dim = 11
        self.fuse_net = fusion_net(in_dim=[38,64,64,192],hidden_dim=64,latent_dim=48,T=10,xyz_dim=self.in_dim,feature_dim=64)
        self.xyz_decoder = nn.Sequential(nn.ReLU(),nn.Linear(64,64),
                                          nn.ReLU(),nn.Linear(64,3))
        self.rotation_decoder = nn.Sequential(nn.ReLU(),nn.Linear(64,64),
                                          nn.ReLU(),nn.Linear(64,4))
        self.scale_decoder = nn.Sequential(nn.ReLU(),nn.Linear(64,64),
                                          nn.ReLU(),nn.Linear(64,3))

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)


    def encode_audio(self, a):
        if a is None: return None
        enc_a = self.audio_net(a)  # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # [1, 64]

        return enc_a

    def forward(self, x, a, e, auds_exp, audio_style, audio_emotion, stage=None):
        enc_x = self.xyz_encoder(x)
        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att
        eye_att = torch.relu(self.eye_att_net(enc_x))
        enc_e = self.exp_encode_net(e[:-1])
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        enc_e = enc_e * eye_att
        tmp_auds_exp = auds_exp.repeat(enc_x.shape[0], 1)
        tmp_audio_style = audio_style.repeat(enc_x.shape[0], 1)
        tmp_audio_emotion = self.emotion_encoder(x, audio_emotion)

        fuse_W = self.fuse_net(torch.cat([enc_w, enc_e],dim=-1),tmp_auds_exp,tmp_audio_style,tmp_audio_emotion, enc_x, stage)
        d_xyz = self.xyz_decoder(fuse_W)
        d_xyz = d_xyz * 1e-2
        d_rot = self.rotation_decoder(fuse_W)
        d_scale = self.scale_decoder(fuse_W)
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': None,
            'd_scale': d_scale,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True),
        }

    def get_params(self, lr, lr_net, wd=0):
        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.xyz_encoder.parameters(), 'lr': lr},
            {'params': self.fuse_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.xyz_decoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.rotation_decoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.scale_decoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.emotion_encoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        return params



class MouthMotionNetwork(nn.Module):
    def __init__(self,audio_dim = 32):
        super(MouthMotionNetwork, self).__init__()
        self.audio_in_dim = 29
        self.bound = 0.15
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)
        self.num_levels = 12
        self.level_dim = 1
        planeconfig = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 3,
            'output_coordinate_dim': 16,
            'resolution': [64, 64, 64]
        }
        multires = [1, 2, 4, 8]
        self.xyz_encoder =HexPlaneField(self.bound,planeconfig,multires)   #对于不同scale拼接，对于同scale的不同平面连乘
        self.in_dim = planeconfig['output_coordinate_dim'] * len(multires)
        self.num_layers = 3
        self.hidden_dim = 32
        self.fuse_net = fusion_net_mouth(in_dim=[32,64,64],hidden_dim=64,latent_dim=48,T=10,xyz_dim=self.in_dim,feature_dim=64)
        self.xyz_decoder = nn.Sequential(nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)


    def encode_audio(self, a):
        if a is None: return None
        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
        return enc_a


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[: ,:1], x[: ,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)


    def forward(self, x, a, auds_exp, audio_style, stage=None):
        enc_x = self.xyz_encoder(x)
        enc_a = self.encode_audio(a)
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        tmp_auds_exp = auds_exp.repeat(enc_x.shape[0], 1)
        tmp_audio_style = audio_style.repeat(enc_x.shape[0], 1)

        fuse_W = self.fuse_net(enc_w,tmp_auds_exp,tmp_audio_style,enc_x,stage)
        d_xyz = self.xyz_decoder(fuse_W)
        d_xyz = d_xyz * 1e-2
        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5
        return {
            'd_xyz': d_xyz
        }


    def get_params(self, lr, lr_net, wd=0):
        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.xyz_encoder.parameters(), 'lr': lr},
            {'params': self.fuse_net.parameters(), 'lr': lr},
            {'params': self.xyz_decoder.parameters(), 'lr': lr},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        return params
