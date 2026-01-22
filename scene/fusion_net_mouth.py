import torch
import torch.nn as nn
import random

class uncertainty_net(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(uncertainty_net, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        mu = self.fc_mu(hidden)
        var = self.fc_var(hidden)

        return mu, var
def batch_covariance(X):
    B, N, D = X.shape
    mean = X.mean(dim=1, keepdim=True)
    X_centered = X - mean
    X_centered_T = X_centered.transpose(1, 2)
    cov = X_centered_T @ X_centered / (N - 1)
    return cov


class fusion_net_mouth(nn.Module):
    def __init__(self, in_dim:list, hidden_dim:int, latent_dim:int, T:int, xyz_dim:int, feature_dim:int):
        super(fusion_net_mouth, self).__init__()
        assert len(in_dim) == 3
        self.latent_dim = latent_dim
        self.T = T
        self.audio_net = nn.ModuleList([uncertainty_net(in_dim[0] + xyz_dim, hidden_dim, latent_dim) for _ in range(T)])
        self.content_net = nn.ModuleList([uncertainty_net(in_dim[1] + xyz_dim, hidden_dim, latent_dim) for _ in range(T)])
        self.style_net = nn.ModuleList([uncertainty_net(in_dim[2] + xyz_dim, hidden_dim, latent_dim) for _ in range(T)])

        self.fc = nn.Linear(latent_dim + xyz_dim, feature_dim)


    def forward(self, audio, content, style, xyz, stage):
        audio = torch.cat([audio, xyz], dim=-1)
        content = torch.cat([content, xyz], dim=-1)
        style = torch.cat([style, xyz], dim=-1)

        if stage == 1:
            model_index = [random.randint(0,9) for _ in range(3)]
            audio_mu, audio_var = self.audio_net[model_index[0]](audio)
            content_mu, content_var = self.content_net[model_index[1]](content)
            style_mu, style_var = self.style_net[model_index[2]](style)
            audio_var = torch.diag_embed(torch.exp(audio_var))
            content_var = torch.diag_embed(torch.exp(content_var))
            style_var = torch.diag_embed(torch.exp(style_var))

        elif stage == 2:
            audio_mu = []
            content_mu = []
            style_mu = []
            audio_var = []
            content_var = []
            style_var = []
            for i in range(self.T):
                tmp = self.audio_net[i](audio)
                audio_mu.append(tmp[0])
                audio_var.append(tmp[1])

                tmp = self.content_net[i](content)
                content_mu.append(tmp[0])
                content_var.append(tmp[1])

                tmp = self.style_net[i](style)
                style_mu.append(tmp[0])
                style_var.append(tmp[1])

            audio_mu = torch.stack(audio_mu, dim=0).transpose(1, 0)
            content_mu = torch.stack(content_mu, dim=0).transpose(1, 0)
            style_mu = torch.stack(style_mu, dim=0).transpose(1, 0)
            audio_var = torch.exp(torch.stack(audio_var, dim=0)).mean(dim=0,keepdim=False)
            content_var = torch.exp(torch.stack(content_var, dim=0)).mean(dim=0,keepdim=False)
            style_var = torch.exp(torch.stack(style_var, dim=0)).mean(dim=0,keepdim=False)
            assert audio_var.shape == torch.Size([xyz.shape[0],self.latent_dim])
            audio_var = batch_covariance(audio_mu) + torch.diag_embed(audio_var)
            content_var = batch_covariance(content_mu) + torch.diag_embed(content_var)
            style_var = batch_covariance(style_mu) + torch.diag_embed(style_var)
            audio_mu = audio_mu.mean(dim=1, keepdim=False)
            content_mu = content_mu.mean(dim=1, keepdim=False)
            style_mu = style_mu.mean(dim=1, keepdim=False)

        else:
            raise RuntimeError('Wrong stage!')

        assert audio_mu.shape == torch.Size([xyz.shape[0],self.latent_dim])
        assert audio_var.shape == torch.Size([xyz.shape[0],self.latent_dim,self.latent_dim])

        audio_var_inverse = torch.inverse(audio_var)
        content_var_inverse = torch.inverse(content_var)
        style_var_inverse = torch.inverse(style_var)
        fused_var = torch.inverse(audio_var_inverse + content_var_inverse + style_var_inverse)
        tmp = (torch.bmm(audio_var_inverse, audio_mu.unsqueeze(2)) +
               torch.bmm(content_var_inverse, content_mu.unsqueeze(2)) +
               torch.bmm(style_var_inverse, style_mu.unsqueeze(2)))
        fused_mu = torch.bmm(fused_var, tmp).squeeze(2)
        h = torch.cat([fused_mu,xyz],dim=-1)
        return self.fc(h)
