# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch.nn as nn
import torch

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

from .google import JointAutoregressiveHierarchicalPriors


class Cheng2020Anchor(JointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2), #/2: 128
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2), # /2:64
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),# /2: 32
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2), # /2 : 16
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N), # do not change size: 16
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N), # do not change size: 16
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2), # /2: 8
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2), # /2: 4
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class Cheng2020Attention(Cheng2020Anchor):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )


class Cheng2020Anchor_Transfer(Cheng2020Anchor):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2), #/2: 128
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2), # /2:64
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),# /2: 32
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2), # /2 : 16
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N), # do not change size: 16
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N), # do not change size: 16
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2), # /2: 8
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2), # /2: 4
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.m_a_time = nn.Sequential(
            nn.Linear(128, N),
            nn.LeakyReLU(inplace=True),
            nn.Linear(N, N),
        )

        self.m_a = nn.Sequential(
            conv3x3(N+8, N), # do not change size: 16
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N), # do not change size: 16
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N), # /2: 8
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N), # /2: 4
        )
        self.n_a =  nn.Sequential(
            conv3x3(N+8, N), # do not change size: 16
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N), # do not change size: 16
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N), # /2: 8
        )
        self.mask_pooling_y = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.AvgPool2d(2, stride=2),
            nn.AvgPool2d(2, stride=2),
            nn.AvgPool2d(2, stride=2),
        )
        self.mask_pooling_z = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.AvgPool2d(2, stride=2),
        )

    def conditional_mapping(self, x, target_x, x_time_embedding, target_x_time_embedding, loc_mask):
        print('mapping')
        print(x.shape)
        y = self.g_a(x)
        z = self.h_a(y)

        target_y =self.g_a(target_x)
        target_z = self.h_a(target_y)

        concat_time_embedding = torch.cat([x_time_embedding, target_x_time_embedding], dim=1)
        concat_time_embedding = self.m_a_time(concat_time_embedding) # (bs, N)
        concat_time_embedding = concat_time_embedding.unsqueeze(-1).unsqueeze(-1) # (bs, N, 1, 1)

        print(loc_mask.shape)
        loc_mask_y = self.mask_pooling_y(loc_mask)
        loc_mask_z = self.mask_pooling_z(loc_mask_y)
        print(loc_mask_y.shape)
        print(loc_mask_z.shape)
        y = y + concat_time_embedding
        z = z + concat_time_embedding
        concat_y = torch.cat([y, loc_mask_y], dim=1)
        concat_z = torch.cat([z, loc_mask_z], dim=1)
        hat_target_y = self.m_a(concat_y)
        hat_target_z = self.n_a(concat_z)
        print(concat_y.shape, concat_z.shape)
        print(hat_target_y.shape, hat_target_z.shape)
        # sleep

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = (z_hat + hat_target_z) * 0.5
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        # gaussian_params.shape = (1, 384, 16, 16) if patch H and W = 256
        # scales_hat.shape = (1, 192, 16, 16)
        # means_hat.shape = (1, 192, 16, 16)
        scales_hat, means_hat = gaussian_params.chunk(2, 1) 
        # print(gaussian_params.shape)
        # print(scales_hat.shape, means_hat.shape)
        # sleep
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat * 0.5 + hat_target_y * 0.5
        # torch.zeros(5).uniform_(-0.5,0.5) = torch.
        x_hat = self.g_s(y_hat)
        print(x_hat.shape)
        # plt.imshow(x_hat[0][0].cpu().detach().numpy())
        # plt.title("cheng2020 x_hat after g_s")
        # plt.savefig('x_hat after g_s.png')
        # exit()
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "target_y": target_y,
            "target_z": target_z,
            "hat_target_y": hat_target_y,
            "hat_target_z": hat_target_z,
        }
        
    def forward(self, x, target_x=None, x_time_embedding=None, target_x_time_embedding=None, loc_mask=None, loc_mask_target=None, flag=0):
        if flag == 1:
            return self.conditional_mapping(x, target_x, x_time_embedding, target_x_time_embedding, loc_mask)
        elif flag == 2:
            return self.conditional_prediction(x, target_x, x_time_embedding, target_x_time_embedding, loc_mask, loc_mask_target)
        
        else:
            # print(x.shape)
            # plt.imshow(x[0][0].cpu().detach().numpy())
            # plt.title("cheng2020 input x")
            # plt.colorbar()
            # plt.savefig('x_input.png')
            # print(i)
            # print(x.shape)
            y = self.g_a(x)
            # print(y.shape)
            # y_name = 'g_a_1_192_48_92_1'
            # np.save(y_name, y.cpu().detach().numpy())
            # plt.imshow(y.cpu().detach().numpy())
            # plt.title("cheng2020 after g_a x")
            # plt.savefig('x after g_a.png')

            z = self.h_a(y)
            # print(z.shape)
            # z_name = 'h_a_63_192_12_23_1'
            # np.save(z_name, z.cpu().detach().numpy())
            # return y, z
            # exit()

            # CVAE ENCODER DECODER (m_a, m_s)
            # ENC: P(U(different from z in hyperprior)|C,y_hat,z_hat), DEC: P(y_hat',z_hat'|U,C')
            # C KNOWN (TIME, Longitude, Latitude)),
            # Position embedding style embedding, originally for BERT position embedding
            # Trend + Periods embedding,
            # Trend -> 1951, 1952,1953, ... -> directly use BERT positional embedding
            # Periods -> {Similar to BERT embedding, but has only fix number}
            # Ex: for month, we need 12, for day we need 31, for hour we need 24
            # Final embedding = Trend + Periods (in case what resolution you want)

            # C = Embedding_year(1951) + Embedding_month(1)
            # C' = Embedding_year(1952) + Embedding_month(12)
            # Enc -> If we want to infer time from an image patch,
            # we could just build a classifier with number of class: # of years in embedding year
            # + # of months in embedding_month,
            # m_a will contain information that will not change across dataset,
            # once we know the anchor metadata and resolution

            # About modality: currently we restrict 3 channels to be mod A, mod B , mod A*B
            # Suppose we already have t, u, t*u, then t can be reused in t * v
            # Fix size of modality embedding, ex: temp, u, v, perticipation, ocean surface temp
            # we could also add u * temp, v * temp and get their embedding.
            # Embedding_modality={'temp': d_temp, 'u': d_u, 'v':..., 'temp*u': d_temp*u}


            # Assume given y_hat, z_hat, if U can be obtained by y_hat and z_hat

            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)

            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )
            ctx_params = self.context_prediction(y_hat)
            # print(params.shape)
            # print(ctx_params.shape)

            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            # gaussian_params.shape = (1, 384, 16, 16) if patch H and W = 256
            # scales_hat.shape = (1, 192, 16, 16)
            # means_hat.shape = (1, 192, 16, 16)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            # print(gaussian_params.shape)
            # print(scales_hat.shape, means_hat.shape)
            # sleep
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

            x_hat = self.g_s(y_hat)
            # print(x_hat.shape)
            # plt.imshow(x_hat[0][0].cpu().detach().numpy())
            # plt.title("cheng2020 x_hat after g_s")
            # plt.savefig('x_hat after g_s.png')
            # exit()
            return {
                "x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            }


    def conditional_prediction(self, x, target_x, x_time_embedding, target_x_time_embedding, loc_mask_anchor, loc_mask_target):
        # print(x.shape)
        y = self.g_a(x)
        z = self.h_a(y)

        target_y =self.g_a(target_x)
        target_z = self.h_a(target_y)
        # print(x_time_embedding.shape, target_x_time_embedding.shape)
        concat_time_embedding = torch.cat([x_time_embedding, target_x_time_embedding], dim=1)
        concat_time_embedding = self.m_a_time(concat_time_embedding) # (bs, N)
        concat_time_embedding = concat_time_embedding.unsqueeze(-1).unsqueeze(-1) # (bs, N, 1, 1)


        loc_mask_anchor_y = self.mask_pooling_y(loc_mask_anchor)
        loc_mask_anchor_z = self.mask_pooling_z(loc_mask_anchor_y)
        # loc_mask_target_y = self.mask_pooling_y(loc_mask_target)
        # loc_mask_target_z = self.mask_pooling_z(loc_mask_target_z)
        
        # print(loc_mask_y.shape)
        # print(loc_mask_z.shape)
        y = y + concat_time_embedding
        z = z + concat_time_embedding
        # print(loc_mask_anchor_y.shape, loc_mask_anchor_z.shape, y.shape, z.shape)
        concat_y = torch.cat([y, loc_mask_anchor_y], dim=1)
        concat_z = torch.cat([z, loc_mask_anchor_z], dim=1)
        hat_target_y = self.m_a(concat_y)
        hat_target_z = self.n_a(concat_z)
        # print(concat_y.shape, concat_z.shape)
        # print(hat_target_y.shape, hat_target_z.shape)
        # sleep

        target_z_hat, target_z_likelihoods = self.entropy_bottleneck(target_z)

        target_params = self.h_s(target_z_hat)

        target_y_hat = self.gaussian_conditional.quantize(
            target_y, "noise" if self.training else "dequantize"
        )
        target_ctx_params = self.context_prediction(target_y_hat)
        target_gaussian_params = self.entropy_parameters(
            torch.cat((target_params, target_ctx_params), dim=1)
        )
        # sleep
        # gaussian_params.shape = (1, 384, 16, 16) if patch H and W = 256
        # scales_hat.shape = (1, 192, 16, 16)
        # means_hat.shape = (1, 192, 16, 16)
        target_scales_hat, target_means_hat = target_gaussian_params.chunk(2, 1) 
        # print(gaussian_params.shape)
        # print(scales_hat.shape, means_hat.shape)
        # sleep
        _, target_y_likelihoods = self.gaussian_conditional(target_y, target_scales_hat, means=target_means_hat)
        target_x_hat = self.g_s(target_y_hat)

        z_hat, z_likelihoods = self.entropy_bottleneck(hat_target_z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            hat_target_y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        # gaussian_params.shape = (1, 384, 16, 16) if patch H and W = 256
        # scales_hat.shape = (1, 192, 16, 16)
        # means_hat.shape = (1, 192, 16, 16)
        scales_hat, means_hat = gaussian_params.chunk(2, 1) 
        # print(gaussian_params.shape)
        # print(scales_hat.shape, means_hat.shape)
        # sleep
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat = self.g_s(y_hat)
        # print(x_hat.shape)
        # plt.imshow(x_hat[0][0].cpu().detach().numpy())
        # plt.title("cheng2020 x_hat after g_s")
        # plt.savefig('x_hat after g_s.png')
        # exit()
        return {
            "x_hat": x_hat,
            "target_x_hat": target_x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "target_likelihoods":{ "target_y": target_y_likelihoods, "target_z": target_z_likelihoods},
            "target_y_hat": target_y_hat,
            "target_z_hat": target_z_hat,
            "hat_target_y": hat_target_y,
            "hat_target_z": hat_target_z,
        }


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net
