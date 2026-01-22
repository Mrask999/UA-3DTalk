from scene.hexplane import HexPlaneField
import torch
import math
import torch
import torch.nn as nn


class EmotionEncoder(nn.Module):
    def __init__(self,bound, planeconfig, multires):
        super(EmotionEncoder, self).__init__()

        self.bound = bound
        self.planeconfig = planeconfig
        self.multires = multires

        assert self.planeconfig['output_coordinate_dim'] == 64, self.planeconfig['output_coordinate_dim']

        self.emotion_fc1 = nn.Linear(128,64)

        self.xyz_encoder = HexPlaneField(self.bound, planeconfig, multires)
        self.in_dim = planeconfig['output_coordinate_dim'] * len(multires)

        self.codebook = nn.ParameterList()

        D = self.planeconfig['output_coordinate_dim']
        self.d_2 = math.sqrt(D)

        for i in range(len(self.multires)):
            H = self.planeconfig['resolution'][i] * self.multires[i]
            tmp = nn.Parameter(torch.randn(H, D), requires_grad=True)
            nn.init.uniform_(tmp, a=0.1, b=0.5)

            self.codebook.append(tmp)

    def query_emotion(self, emotion, codebook):
        attn_score = torch.matmul(emotion, codebook.T) / self.d_2
        assert attn_score.shape[0] == 1, attn_score.shape[0]

        attn_score = torch.softmax(attn_score, dim=1)
        return torch.matmul(attn_score, codebook)


    def forward(self, xyz, emotion):
        emotion_64 = self.emotion_fc1(emotion)
        enc_xyz = self.xyz_encoder(xyz)

        D = self.planeconfig['output_coordinate_dim']
        assert enc_xyz.shape == torch.Size([xyz.shape[0], self.in_dim]), enc_xyz.shape

        multi_scale_interp = []
        for i in range(len(self.multires)):
            tmp_xyz = enc_xyz[:,i*D:(i+1)*D]
            s_emotion = self.query_emotion(emotion_64, self.codebook[i])
            s_emotion = s_emotion.repeat(xyz.shape[0], 1)  #[N, 64]

            multi_scale_interp.append(tmp_xyz * s_emotion)

        multi_scale_interp = torch.cat(multi_scale_interp,dim=-1)
        assert multi_scale_interp.shape[1] == self.in_dim, multi_scale_interp.shape

        return multi_scale_interp







