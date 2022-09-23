import argparse
import math
import random
import shutil
import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import torch.nn.functional as F

from compressai.zoo import image_models
from compressai.datasets import CustomTensorDataset
from torchvision.transforms.functional import crop
from torch.utils.data import ConcatDataset
import torch
import numpy as np
from compressai.layers.layers import PositionalEmbedding  
# import tqdm
# from maskedtensor import masked_tensor
# from maskedtensor import as_masked_tensor
# p_min = [-0.81310266, -0.0005704858, -26.1313]
# max_min = [28.265799+0.81310266, 35.88579+0.0005704858, 825.0345+26.1313]


def time_index_to_year_month(all_time_index):
    all_year = all_time_index // 12
    all_month = all_time_index % 12
    return all_year, all_month

def fetch_time_embedding_year_month(time_year_embedding, time_month_embedding, year, month, time_dim=64):
    # print(time_month_embedding.pe[0, year,:].shape)
    return time_year_embedding.pe[0, year, :] + time_month_embedding.pe[0, month, :]

def batch_time_embedding(time_year_embedding, time_month_embedding, all_year, all_month, time_dim=64):
    time_embedding = torch.zeros(all_year.shape[0], time_dim)
    for i in range(all_year.shape[0]):
        time_embedding[i, :] = fetch_time_embedding_year_month(time_year_embedding, time_month_embedding, all_year[i], all_month[i], time_dim)
    return time_embedding

def fetch_time_embedding(time_index, all_time_index_embedding):
    return all_time_index_embedding[time_index, :]

def construct_location_embedding_padding(long_embedding, lati_embedding, left, top, lat_res=721, long_res=1440, lat_dim=4, long_dim=4, padding_direction=[16, 16, 23, 24]):
    pad_left, pad_right, pad_top, pad_bottom = padding_direction
    pad_col = pad_left + pad_right
    pad_row = pad_top + pad_bottom
    mask = torch.zeros((1, lat_res + pad_row, long_res + pad_col, long_dim + lat_dim))
    padding_embed = torch.cat([lati_embedding.pe[0, 0, :], long_embedding.pe[0, 0, :]]).reshape(1, 1, 1, -1)
    # print(padding_embed)
    mask = mask + padding_embed
    # print(mask, mask.shape)
    for i in range(left + pad_left, left + pad_left + lat_res):
        for j in range(top + pad_top, top + pad_top + long_res):
            # print(i, j)
            mask[0, i, j, :] = torch.cat([lati_embedding.pe[0, i - left - pad_left + 1, :], long_embedding.pe[0, j - top - pad_top + 1, :]])
    # print(mask, mask.shape)
    # sleep
    return mask

def batch_location_embedding(grid_location_embedding_padding, left, top, lat_res=721, long_res=1440, loc_res=8):
    bs = left.size(0)
    mask = torch.cat([grid_location_embedding_padding[:, left[i]: left[i] + lat_res, top[i]: top[i] + long_res, :] for i in range(bs)], dim=0)
    # print(mask)
    return mask




def fetch_target_data(time_index, dataset_index, left_list, top_list, concat_dataset_train, lati_res=721, long_res=1440, total_time=756, padding=[16, 16, 23, 24]):
    sampling_method = ["same month", "neighbour time", "random"]
    sampling_method = ["same month"]
    bs = len(time_index)
    pad_left, pad_right, pad_top, pad_bottom = padding
    len_sampling_method = len(sampling_method)
    sample_list = torch.randint(0, len_sampling_method, size=(bs,))
    new_time_index = torch.zeros_like(time_index)
    for i in range(bs):
        if sampling_method[sample_list[i]] == "same month":
            new_time_index[i] = (time_index[i] + 12) % total_time
            # new_time_index[i] = (time_index[i] + torch.randint(1, total_time // 12, size=(1,)).item() * 12) % total_time
        elif sampling_method[sample_list[i]] == "neighbour time":
            new_time_index[i] = time_index[i] + torch.randint(-2, 2, size=(1,)).item()
        elif sampling_method[sample_list[i]] == "random":
            new_time_index[i] = torch.randint(0, total_time, size=(1,)).item()
        else:
            raise ValueError("Wrong sampling method")
    target_data = torch.cat([concat_dataset_train.datasets[dataset_index[i]][new_time_index[i]][0].unsqueeze(0) for i in range(bs)], dim=0)
    # print(target_data.shape)
    target_data = F.pad(
        target_data,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )
    # print(target_data.shape)
    target_data = torch.cat([target_data[i, :, left_list[i]: left_list[i] + lati_res, top_list[i]: top_list[i] + long_res].unsqueeze(0) for i in range(bs)], dim=0)
    return target_data, new_time_index



time_year_embedding = PositionalEmbedding(d_model=64, max_len=64)
time_month_embedding = PositionalEmbedding(d_model=64, max_len=12, constant=8888)
lati_embedding = PositionalEmbedding(d_model=4, max_len=722, constant=7777)
long_embedding = PositionalEmbedding(d_model=4, max_len=1441, constant=6666)

# grid_location_embedding_padding = construct_location_embedding_padding(long_embedding, lati_embedding, 0, 0, long_res=3, lat_res=3, long_dim=4, lat_dim=4,padding_direction=[1,2,3,4])
# batch_location_embedding(grid_location_embedding_padding, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), lat_res=4, long_res=4, loc_res=8)
# sleep
all_time_index = torch.arange(0, 756)
all_year, all_month = time_index_to_year_month(all_time_index)
all_time_index_embedding = batch_time_embedding(time_year_embedding, time_month_embedding, all_year, all_month, time_dim=64)


grid_location_embedding_padding = construct_location_embedding_padding(long_embedding, lati_embedding, 0, 0, lat_res=721, long_res=1440, long_dim=4, lat_dim=4, padding_direction=[16, 16, 23, 24])

def crop256(image):
    i = torch.randint(0, 1302 - 256 + 1, size=(1,)).item()
    j = torch.randint(0, 300 - 256 + 1, size=(1,)).item()
    # print("i:", i)
    # print("j:", j)
    return crop(image, i, j, 256, 256)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        # print(out["mse_loss"])
        # print(output["x_hat"][0, 0, :, :])
        # print(output["x_hat"][0, 1, :, :])
        # print(out["mse_loss"])
        # print(target[0, 0, :, :])
        # print(target[0, 1, :, :])
        # exit()
        out["loss"] = self.lmbda * 255 **2 * out["mse_loss"] + out["bpp_loss"]

        out["temp_mse"] = self.mse(output["x_hat"][:, 0, :, :], target[:, 0, :, :])
        out["salt_mse"] = self.mse(output["x_hat"][:, 1, :, :], target[:, 1, :, :])
        out["zeta_mse"] = self.mse(output["x_hat"][:, 2, :, :], target[:, 2, :, :])
        ts = torch.mul(output["x_hat"][:, 0, :, :], output["x_hat"][:, 1, :, :])
        ts_hat = torch.mul(target[:, 0, :, :], target[:, 1, :, :])
        out["temp_salt_mse"] = self.mse(ts, ts_hat)
        return out

class RateDistortionLoss_cond_mapping(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        # out["predict_z_loss"] = 0
        # out["predict_y_loss"] = 0
        # out["predict_same_z_loss"] = 0
        # out["predict_same_y_loss"] = 0

        out["predict_z_loss"] = self.mse(output["target_z"], output["target_z"])
        out["predict_y_loss"] = self.mse(output["target_y"], output["target_y"])

        out["predict_same_z_loss"] = self.mse(output["z"], output["z"])
        out["predict_same_y_loss"] = self.mse(output["y"], output["y"])

        out["loss"] = out["bpp_loss"] # self.lmbda * 255 **2 * (out["mse_loss"]) #  + 0.1 * (out["predict_y_loss"] + out["predict_z_loss"]) + 0.1 * (out["predict_same_y_loss"] + out["predict_same_z_loss"])) + out["bpp_loss"]

        out["temp_mse"] = self.mse(output["x_hat"][:, 0, :, :], target[:, 0, :, :])
        out["salt_mse"] = self.mse(output["x_hat"][:, 1, :, :], target[:, 1, :, :])
        out["zeta_mse"] = self.mse(output["x_hat"][:, 2, :, :], target[:, 2, :, :])
        ts = torch.mul(output["x_hat"][:, 0, :, :], output["x_hat"][:, 1, :, :])
        ts_hat = torch.mul(target[:, 0, :, :], target[:, 1, :, :])
        out["temp_salt_mse"] = self.mse(ts, ts_hat)
        return out






def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    print(len(union_params), len(inter_params), len(params_dict))
    print(next(net.g_a.parameters()).requires_grad, next(net.m_a.parameters()).requires_grad)
    assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, concat_dataset_train=None, log_dir=None
):
    model.train()
    device = next(model.parameters()).device

    loss_list = AverageMeter()
    bpp_loss_list = AverageMeter()
    mse_loss_list = AverageMeter()
    aux_loss_list = AverageMeter()
    temp_mse_list = AverageMeter()
    salt_mse_list = AverageMeter()
    temp_salt_mse_list = AverageMeter()
    zeta_mse_list =AverageMeter()

    mse_target_y_loss_list = AverageMeter()
    mse_target_z_loss_list = AverageMeter()    

    mse_same_y_loss_list = AverageMeter()
    mse_same_z_loss_list = AverageMeter()  

    zero_counts_distribution = torch.zeros(12)
    padding = [16, 16, 23, 24]
    pad_left, pad_right, pad_top, pad_bottom = padding
    idx_left, idx_right, idx_top, idx_bottom = [23, 24, 16, 16]
    for i, d in enumerate(train_dataloader):
        d, time_index, dataset_index = d
        d = d.float() # change to float 32 to match the model
        bs, c, h, w = d.size(0), d.size(1), d.size(2), d.size(3) # batch size, channel, height(721), width(1440)
        p = 64  # maximum 6 strides of 2
        d = F.pad(
            d,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )
        # print(d.shape)
        # new_h = (h + p - 1) // p * p
        # new_w = (w + p - 1) // p * p
        # padding_left = (new_w - w) // 2
        # padding_right = new_w - w - padding_left
        # padding_top = (new_h - h) // 2
        # padding_bottom = new_h - h - padding_top
        # d = F.pad(
        #     d,
        #     (padding_left, padding_right, padding_top, padding_bottom),
        #     mode="constant",
        #     value=0,
        # )



        # h, w = d.size(2), d.size(3)
        patch_sizes = [64, 128, 256]
        patch_sizes = [256]
        patch_size = np.random.choice(patch_sizes, 1)[0]
        # # print(h-patch_size+1)

        # left = torch.randint(0, h-patch_size+1+idx_left, size=(1,)).item()
        # top = torch.randint(0, w-patch_size+1+idx_top, size=(1,)).item()
        left_list = torch.randint(0, h-patch_size+1+idx_left, size=(bs,))
        top_list = torch.randint(0, w-patch_size+1+idx_top, size=(bs,))
        # print(left_list)
        # print(top_list)
        # print(left, top)
        # sleep

        # right_list = left_list + patch_size
        # bottom_list = top_list + patch_size
        
        # d = d[:, :, left:right, top:bottom]
        # d = crop(d, left,top, patch_size, patch_size)
        loc_emb = batch_location_embedding(grid_location_embedding_padding, left_list, top_list, lat_res=patch_size, long_res=patch_size, loc_res=8)
        loc_emb = torch.permute(loc_emb, (0, 3, 1, 2))
        # print(loc_emb.shape)
        # sleep
        d = torch.cat([d[i, :, left_list[i]: left_list[i] + patch_size, top_list[i]: top_list[i] + patch_size].unsqueeze(0) for i in range(bs)], dim=0)
        # print(d.shape)
        target_data, new_time_index = fetch_target_data(time_index, dataset_index, left_list, top_list, concat_dataset_train, lati_res=patch_size, long_res=patch_size, total_time=756, padding=padding)
        # print(target_data.shape)
        d_time_embedding = fetch_time_embedding(time_index, all_time_index_embedding)
        target_data_time_embedding = fetch_time_embedding(new_time_index, all_time_index_embedding)        
        # print(d_time_embedding.shape)
        # print(target_data_time_embedding.shape)
        # sleep
        d = d.float().to(device)
        target_data = target_data.float().to(device)
        d_time_embedding = d_time_embedding.float().to(device)
        target_data_time_embedding = target_data_time_embedding.float().to(device)
        loc_emb = loc_emb.float().to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d, target_x=target_data, x_time_embedding=d_time_embedding, target_x_time_embedding=target_data_time_embedding, loc_mask=loc_emb, flag=3)
        # print(d[0, 0, :, :])
        # sleep

        # n = d.cpu().numpy().flatten()
        # print(np.count_nonzero(n > 1))
        # plt.hist(n)
        # plt.show()

        # n = d[:, 1 ,:, :].cpu().numpy()
        # print(np.where(n == 1))
        # print(torch.max(d[:, 1, :, :]))
        # print(torch.min(d))
        # print((d == 1).nonzero().squeeze())
        # exit()

        # d = torch.nan_to_num(d, posinf=-1)
        # !zero grad:
            # optimizer.zero_grad()
            # aux_optimizer.zero_grad()
        # print(d.shape)
        # scale_100 = [3.4669, 2.7866, 11.7793]
        # print(d[0, 0, :, :])
        # print(d[0, 1, :, :])
        # print(d[0, 2, :, :])
        # for i in range(3):


        # d[:, 0, :, :] = torch.where(d[:, 0, :, :] == 3.4669, 0, d[:, 0, :, :]+0.1)
        # print(d[0, 0, :, :])
        # print(d[0, 1, :, :])
        # print(d[0, 2, :, :])

        # exit()
        # d = d + 0.1
        # d = torch.add(d, 0.1)
        # d[d > 2] = 0

        # print(d[0, 0, :, :])
        # print(d[0, 0, 0, 0])
        # print(d[0, 1, :, :])
        # print(d[0, 2, :, :])

        # print(torch.max(d[:, 0, :, :]))
        # print(torch.min(d))
        # exit()
        # print(d.shape)
        # with torch.no_grad():
        #     y, z = model(d)
        # y_name = 'g_a_'+str(d.shape[0])+'_192_48_92_' + str(i)
        # np.save(y_name, y.cpu().detach().numpy())
        #
        # z_name = 'h_a_16_192_12_23_' + str(i)
        # np.save(z_name, z.cpu().detach().numpy())
        # print(i)
        #
        # continue
        # out_net['x_hat'].clamp_(0, 1)
        # print(out_net['x_hat'][0, 2, :, :])
        # print(out_net['x_hat'][0, 0, :, :])
        # exit()

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        bpp_loss_list.update(out_criterion["bpp_loss"].detach())
        loss_list.update(out_criterion["loss"].detach())
        mse_loss_list.update(out_criterion["mse_loss"].detach())
        aux_loss_list.update(aux_loss.detach())
        temp_mse_list.update(out_criterion["temp_mse"].detach())
        salt_mse_list.update(out_criterion["salt_mse"].detach())
        zeta_mse_list.update(out_criterion["zeta_mse"].detach())
        temp_salt_mse_list.update(out_criterion["temp_salt_mse"].detach())

        mse_target_y_loss_list.update(out_criterion["predict_y_loss"].detach())
        mse_target_z_loss_list.update(out_criterion["predict_z_loss"].detach())
        mse_same_y_loss_list.update(out_criterion["predict_same_y_loss"].detach())
        mse_same_z_loss_list.update(out_criterion["predict_same_z_loss"].detach())


        if i % 10 == 0:
            # print(i)
            log(log_dir / 'log.txt',f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tMSE y loss: {out_criterion["predict_y_loss"].item():.6f} |'
                f'\tMSE z loss: {out_criterion["predict_z_loss"].item():.6f} |'
                f'\tMSE same y loss: {out_criterion["predict_same_y_loss"].item():.6f} |'
                f'\tMSE same z loss: {out_criterion["predict_same_z_loss"].item():.6f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}")
            # plt.plot(zero_counts_distribution)

            # for x, y in zip(np.arange(12), zero_counts_distribution):
            #     plt.text(x=x,y=y, s=str(int(y.item())))
            # file_name = "epoch"+str(epoch)+"batch"+str(i//10)
            # plt.show()
            #
            # plt.title("zero number percent for 10 batches")
            # plt.savefig(file_name)
            # plt.clf()
            # if i != 0:
            #     zero_counts_distribution = torch.zeros(12)

            # print(
            #     f"Train epoch {epoch}: ["
            #     f"{i*len(d)}/{len(train_dataloader.dataset)}"
            #     f" ({100. * i / len(train_dataloader):.0f}%)]"
            #     f'\tLoss: {out_criterion["loss"].item():.3f} |'
            #     f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
            #     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
            #     f"\tAux loss: {aux_loss.item():.2f}"
            # )
    log(log_dir / 'log.txt', f"Train epoch {epoch}: Average losses:"
          f"\tLoss: {loss_list.avg:.3f} |"
          f"\tMSE loss: {mse_loss_list.avg:.8f} |"
          f'\tMSE y loss: {mse_target_y_loss_list.avg:.6f} |'
          f'\tMSE z loss: {mse_target_z_loss_list.avg:.6f} |'
          f'\tMSE same y loss: {mse_same_y_loss_list.avg:.6f} |'
          f'\tMSE same z loss: {mse_same_z_loss_list.avg:.6f} |'
          f"\tBpp loss: {bpp_loss_list.avg:.2f} |"
          f"\tAux loss: {aux_loss_list.avg:.2f}")
    # print(mse_loss_list)
    print('epoch {} train loss: {}'.format(epoch, loss_list.avg))
    print('learning rate: {}'.format(optimizer.param_groups[0]['lr']))
    print('mse loss:', mse_loss_list.avg)
    print('bpp loss:', bpp_loss_list.avg)
    print('target y loss:', mse_target_y_loss_list.avg)
    print('target z loss:', mse_target_z_loss_list.avg)
    print('target same y loss:', mse_same_y_loss_list.avg)
    print('target same z loss:', mse_same_z_loss_list.avg)
    # exit()
    return loss_list.avg.item(), mse_loss_list.avg.item(), temp_mse_list.avg.item(), salt_mse_list.avg.item(), zeta_mse_list.avg.item(), temp_salt_mse_list.avg.item()


def test_epoch(epoch, test_dataloader, model, criterion, log_dir=None):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    temp_mse_list = AverageMeter()
    salt_mse_list = AverageMeter()
    zeta_mse_list = AverageMeter()
    temp_salt_mse_list = AverageMeter()

    # mse_y_loss = AverageMeter()
    # mse_z_loss = AverageMeter()
    y_list = []
    z_list = []
    padding = [16, 16, 23, 24]
    pad_left, pad_right, pad_top, pad_bottom = padding   
    with torch.no_grad():
        for d in test_dataloader:
            d, time_index, dataset_index = d
            print(time_index)
            d = d.float()
            d = F.pad(
                d,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )

            # d = d + 0.1
            # d[d > 1.1] = 0
            d = d.to(device)

            # d = torch.nan_to_num(d, posinf=-1)
            # print(d[0, 0, :, :])
            out_net = model(d)
            y, z = out_net
            y = y.cpu()
            z = z.cpu()
            torch.cuda.empty_cache()
            y_list.append(y)
            z_list.append(z)
            continue
            # print(out_net[0, 0, :, :])
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"].detach())
            loss.update(out_criterion["loss"].detach())
            mse_loss.update(out_criterion["mse_loss"].detach())
            # mse_y_loss.update(out_criterion["predict_y_loss"].detach())
            # mse_z_loss.update(out_criterion["predict_z_loss"].detach())
            temp_mse_list.update(out_criterion["temp_mse"].detach())
            salt_mse_list.update(out_criterion["salt_mse"].detach())
            zeta_mse_list.update(out_criterion["zeta_mse"].detach())
            temp_salt_mse_list.update(out_criterion["temp_salt_mse"].detach())
    new_y = torch.cat(y_list, dim=0)
    new_z = torch.cat(z_list, dim=0)
    print(new_y.shape)
    print(new_z.shape)
    np.save('/efs/users/zucksliu/env_project/wind_dataset/wind_y_192_48_92.npy', new_y.numpy())
    np.save('/efs/users/zucksliu/env_project/wind_dataset/wind_z_192_12_23.npy', new_z.numpy())
    sleep

    log(log_dir / 'log.txt', f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.8f} |"
        # f"\tMSE y loss: {mse_y_loss.avg:.3f} |"
        # f"\tMSE z loss: {mse_z_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    print('epoch {} test loss: {}'.format(epoch, loss.avg))
    print('mse loss:', mse_loss.avg)
    print('bpp loss:', bpp_loss.avg)


    return loss.avg.item(), mse_loss.avg.item(), temp_mse_list.avg.item(), salt_mse_list.avg.item(), zeta_mse_list.avg.item(), temp_salt_mse_list.avg.item()



def save_checkpoint(state, is_best, filename="ocean_cheng2020.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "ocean_cheng2020_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices = image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=False, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=800,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.8,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--channel", action="store_true", default =False, help="change the third channel to temp multiply slat")
    parser.add_argument("--dir", type=str, default='./models/cheng2020-anchor-transfer/', help="Exported model directory.")
    args = parser.parse_args(argv)
    return args


def main(argv):
    # print(torch.cuda.device_count())
    # print(torch.cuda.is_available())
    # print("cuda:{}".format(4))
    # x = torch.arange(0, 863226)
    # x = x.view(1302, 663)
    # print(x.shape)
    # print((x == 16865).nonzero(as_tuple=True)[0])
    # y = transforms.Lambda(crop256)(x)
    # print(y)
    # print(y.shape)
    # exit()
    args = parse_args(argv)
    # net = image_models[args.model](quality=6, combine=True)
    net = image_models[args.model](quality=6, combine=True)
    print_net = True
    if print_net:
        print('The network architecture:')
        print(net)
    total_params = sum(
        param.numel() for param in net.parameters()
    )
    for name, param in net.named_parameters():
        print(name, ':', param.shape)
    # sleep
    print('total param #:', total_params)
    trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print('trainable param #:', trainable_params)

    # f_n = "log_epoch" + str(args.epochs) +"_lambda_" + str(args.lmbda)
    # print(f_n)
    # exit()
    # f = open(f_n, 'w')
    # f.write(str(argv)+"\n")
    # f.close()

    if args.dir is not None:
        output_dir = Path(args.dir)
        # print(output_dir)
        # print(Path.cwd())
        # print(Path.cwd() / output_dir)
        output_dir = Path.cwd() / args.dir
        # print(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()
    print('output_dir:', output_dir)
    log(output_dir / 'log.txt', f"Training started with args: {args}")
    # sleep
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    # temp, salt, zeta
    # p_min = [-0.81310266, -0.0005704858, -3.279859]
    # max_min = [28.265799+0.81310266, 35.88579+0.0005704858, 15.006741+3.279859]
    # p_min = [-0.81310266, -0.0005704858, -26.1313] # LiveOcean temp * salt
    # max_min = [28.265799+0.81310266, 35.88579+0.0005704858, 825.0345+26.1313] # LiveOcean temp * salt
    p_min1 = [199.93072509765625, -19.626953125, -4934.843585180133]  # u10
    max_min1 = [315.47837829589844 - 199.93072509765625, 16.193924903869625 + 19.626953125, 3888.185005817384 + 4934.843585180133]  # u10

    p_min2 = [199.93072509765625, -16.156066894531254, -4381.68313198965] #v10
    max_min2 = [315.47837829589844 - 199.93072509765625, 20.362215995788574 + 16.156066894531254, 4969.35201622079 + 4381.68313198965] #v10

    # train_dataset = ImageFolder('/data/zixinl6/Compress/CLIC_pro', split="train", transform=train_transforms)
    # test_dataset = ImageFolder('/data/zixinl6/Compress/CLIC_pro', split="test", transform=test_transforms)
    # train_data = np.load("/data/zixinl6/downscaling/train_float16_7693_3_1302_663.npy")
    # train_data1 = np.load("/data/zixinl6/downscaling/wind_train_tu_681_3_721_1440.npy")
    # train_data2 = np.load("/data/zixinl6/downscaling/wind_train_tv_681_3_721_1440.npy")
    # train_data = np.load("/data/zixinl6/downscaling/train_7693_3_1302_663.npy")
    train_data1 = np.load('/efs/users/zucksliu/env_project/wind_dataset/wind_tu_756_3_721_1440.npy')
    train_data2 = np.load('/efs/users/zucksliu/env_project/wind_dataset/wind_tv_756_3_721_1440.npy')
    # train_data = np.load("/data/zixinl6/downscaling/valid1_92_3_1302_663.npy")
    # test_data = np.load("/data/zixinl6/downscaling/test_float16_876_3_1302_663.npy")
    test_data1 = np.load("/efs/users/zucksliu/env_project/wind_dataset/wind_test_tu_75_3_721_1440.npy")
    test_data2 = np.load("/efs/users/zucksliu/env_project/wind_dataset/wind_test_tv_75_3_721_1440.npy")

    # test_data = np.load("/data/zixinl6/downscaling/test_876_3_1302_663.npy")
    # print(train_data.dtype)
    # tr = torch.tensor(train_data)
    # te = torch.tensor(test_data)

    tr1 = torch.from_numpy(train_data1)
    # tr[tr == 1.0000e+20] = 100
    tr2 = torch.from_numpy(train_data2)

    te1 = torch.from_numpy(test_data1)
    te2 = torch.from_numpy(test_data2)
    # te = torch.from_numpy(test_data)
    # te[te == 1.0000e+20] = 100
    # if args.channel:
    #     tr[:, 2, :, :] = torch.mul(tr[:, 0, :, :], tr[:, 1, :, :])
    #     te[:, 2, :, :] = torch.mul(te[:, 0, :, :], te[:, 1, :, :])
    #     # tr[tr == 10000] = 0
    #     # te[te == 10000] = 0
    #     print("here")
    # te = torch.where(te == 1.0000e+20, 100, te)
    # tr = torch.zeros(1, 3, 3, 3)
    # for i in range(9):
    #     tr[0, 0, i // 3, i % 3] = i+1
    #     tr[0, 1, i // 3, i % 3] = (i+1) * 2
    #     tr[0, 2, i // 3, i % 3] = (i+1) * 3
    # print(tr)
    # p_min = [1, 2, 3]
    # max_min = [8, 16, 24]
    # print(transforms.Normalize(p_min, max_min)(tr))
    # exit()
    # print(tr.shape)

    device = 'cuda:0' if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)

    net = net.to(device)
    net.fix_main_networks()

    train_transforms1 = transforms.Compose(
        [transforms.Normalize(p_min1, max_min1)]
    )
    train_transforms2 = transforms.Compose(
        [transforms.Normalize(p_min2, max_min2)]
    )

    # train_transforms2 = transforms.Compose(
    #     [transforms.Normalize(p_min, max_min), transforms.Resize((256, 256))]
    # )
    # train_transforms3 = transforms.Compose(
    #     [transforms.Normalize(p_min, max_min), transforms.Lambda(crop256)]
    # )

    test_transforms1 = transforms.Compose(
        [transforms.Normalize(p_min1, max_min1), transforms.CenterCrop((256, 256))]
    )
    test_transforms2 = transforms.Compose(
        [transforms.Normalize(p_min2, max_min2), transforms.CenterCrop((256, 256))]
    )

    train_dataset1 = CustomTensorDataset(tr1, transforms = train_transforms1, dataset_index=0)
    train_dataset2 = CustomTensorDataset(tr2, transforms = train_transforms2, dataset_index=1)
    # train_dataset = CustomTensorDataset(tr, transforms = [train_transforms1,train_transforms2, train_transforms3], prob=[10, 30, 60])
    # print(train_dataset.tensors)

    train_dataset = [train_dataset1, train_dataset2]
    concat_dataset_train = ConcatDataset(train_dataset)

    test_dataset1 = CustomTensorDataset(te1, transforms = test_transforms1)
    test_dataset2 = CustomTensorDataset(te2, transforms=test_transforms2)



    train_dataloader = DataLoader(
        concat_dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == device),
    )
    test_dataset = [test_dataset1, test_dataset2]
    # print(train_dataloader.dataset.tensors.dtype)
    concat_dataset_test = ConcatDataset(test_dataset)

    test_dataloader = DataLoader(
        concat_dataset_test,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == device),
    )

    #quality = 6
    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net, device_ids=[0, 1, 2, 3])

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion_test = RateDistortionLoss(lmbda=args.lmbda)
    criterion = RateDistortionLoss_cond_mapping(lmbda=args.lmbda)

    last_epoch = 0
    train_loss = []
    test_loss = []
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        # print(checkpoint["optimizer"])
        # print(optimizer)
        # optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # train_loss = np.load("train_loss.npy").tolist()
        # test_loss = np.load("test_loss.npy").tolist()

    best_loss = float("inf")

    train_mse = []
    temp_list = []
    salt_list = []
    zeta_list = []
    temp_salt_list = []

    t_t_list = []
    s_t_list =[]
    z_t_list = []
    t_s_t_list = []
    test_mse_loss = []
    print('First test loaded model as a reference...')
    log(output_dir / 'log.txt', f"First test loaded model as a reference...")
    # test_epoch(-1, test_dataloader, net, criterion_test, log_dir=output_dir)
    test_epoch(-1, train_dataloader, net, criterion_test, log_dir=output_dir)
    for epoch in range(last_epoch, args.epochs):
        # if epoch == 1:
        #     exit()
        log(output_dir / 'log.txt', f"Learning rate: {optimizer.param_groups[0]['lr']}")
        out_cri, mse, temp_mse, salt_mse, zeta_mse, temp_salt_mse = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            concat_dataset_train,
            log_dir=output_dir
        )


        # train_loss.append(out_cri)
        # train_mse.append(mse)
        # temp_list.append(temp_mse)
        # salt_list.append(salt_mse)
        # zeta_list.append(zeta_mse)
        # temp_salt_list.append(temp_salt_mse)

        # train_mse_loss.append(out_cri['mse_loss'].item())
        loss, mse_t, temp_mse_t, salt_mse_t, zeta_mse_t, temp_salt_mse_t = test_epoch(epoch, test_dataloader, net, criterion_test, log_dir=output_dir)
        lr_scheduler.step(loss)

        test_loss.append(loss)
        test_mse_loss.append(mse_t)
        t_t_list.append(temp_mse_t)
        s_t_list.append(salt_mse_t)
        z_t_list.append(zeta_mse_t)
        t_s_t_list.append(temp_salt_mse_t)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            plt.plot(train_loss, '-o', label='Train')
            plt.plot(test_loss,'-o', label='Test')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['Train', 'Test'])
            plt.title("loss-epoch")
            p_n = output_dir / f"loss-epoch{epoch}.png"
            plt.show()
            plt.savefig(p_n)

            # plt.plot(train_mse, '-o', label = 'train mse')
            # plt.plot(test_mse_loss, '-o',label = 'test mse')
            # plt.xlabel('epoch')
            # plt.ylabel('total mse')
            # plt.legend(['train mse','test mse' ])
            # plt.title("total-mse-epoch")
            # p_n_2 = f"total-mse-epoch{epoch}.png"
            # plt.show()
            # plt.savefig(p_n_2)
            #
            # plt.plot(temp_list, '-o', label = 'temp train mse')
            # plt.plot(t_t_list, '-o', label = 'temp test mse')
            # plt.xlabel('epoch')
            # plt.ylabel('temp mse')
            # plt.legend(['temp train mse', 'temp test mse'])
            # plt.title("temp-mse-epoch")
            # p_n_3 = f"temp-mse-epoch{epoch}.png"
            # plt.show()
            # plt.savefig(p_n_3)


        #save model every 50 epoches and the last epoch:
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print('Saving models...')
            # f_n = f"ocean_quality6_cheng2020_epoch_{epoch}.pth.tar"
            if args.dir is not None:
                output_dir = Path(args.dir)
                Path(output_dir).mkdir(exist_ok=True)
            else:
                output_dir = Path.cwd()
            filename = f"{args.model}_wind_epoch_{epoch}.pth.tar"
            filepath = output_dir / f"{filename}"
            # print(filepath)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "lambda": args.lmbda,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best, filename=filepath
            )
            np.save(output_dir / "train_loss.npy", train_loss)
            np.save(output_dir / "train_mse.npy", train_mse)
            np.save(output_dir / "train_temp_mse.npy", temp_list)
            np.save(output_dir / "train_salt_mse.npy", salt_list)
            np.save(output_dir / "train_zeta_mse.npy", zeta_list)
            np.save(output_dir / "train_temp_salt.npy", temp_salt_list)

            np.save(output_dir / "test_loss.npy", test_loss)
            np.save(output_dir / 'test_total_mse.npy', test_mse_loss)
            np.save(output_dir / "test_temp_mse.npy", t_t_list)
            np.save(output_dir / "test_salt_mse.npy",s_t_list)
            np.save(output_dir / "test_zeta_mse.npy", z_t_list)
            np.save(output_dir / "test_temp_salt_mse.npy", t_s_t_list)
            print('Save model epoch {} to {}'.format(epoch, filepath))





if __name__ == "__main__":
    main(sys.argv[1:])
