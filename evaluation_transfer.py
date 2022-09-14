import math
# import io
import torch
from torchvision import transforms
import numpy as np
# from collections import defaultdict
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from compressai.datasets import CustomTensorDataset
from compressai.utils.eval_model.__main__ import load_checkpoint, collect_images, read_image, eval_model
from examples.train import fetch_time_embedding
from examples.train import construct_location_embedding_padding, batch_location_embedding
from examples.train import batch_time_embedding, time_index_to_year_month
# from examples.train import long_embedding, lati_embedding, fetch_target_data
import argparse
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from compressai.layers.layers import PositionalEmbedding  

time_year_embedding = PositionalEmbedding(d_model=64, max_len=64)
time_month_embedding = PositionalEmbedding(d_model=64, max_len=12, constant=8888)
lati_embedding = PositionalEmbedding(d_model=4, max_len=722, constant=7777)
long_embedding = PositionalEmbedding(d_model=4, max_len=1441, constant=6666)

# grid_location_embedding_padding = construct_location_embedding_padding(long_embedding, lati_embedding, 0, 0, long_res=721, lat_res=1440, long_dim=4, lat_dim=4,padding_direction=[16,16, 23, 24])
all_time_index = torch.arange(0, 756)
all_year, all_month = time_index_to_year_month(all_time_index)
all_time_index_embedding = batch_time_embedding(time_year_embedding, time_month_embedding, all_year, all_month, time_dim=64)


parser = argparse.ArgumentParser()
args = vars(parser.parse_args())

def mse(a, b):
    return torch.mean((a - b)**2).item()

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
metric = 'mse'  # only pre-trained model for mse are available for now
quality = 6    # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)

epoch_100 = load_checkpoint('cheng2020-anchor-transfer', '/efs/users/zucksliu/env_project/compressai/models/cheng2020-anchor-transfer/cheng2020-anchor-transfer_wind_epoch_100-59e07d84.pth.tar')
networks = {
   'epoch_100': epoch_100.to(device),
}
outputs = {}

test_u10 = np.load("/efs/users/zucksliu/env_project/wind_dataset/wind_test_tu_75_3_721_1440.npy")
test_v10 = np.load("/efs/users/zucksliu/env_project/wind_dataset/wind_test_tv_75_3_721_1440.npy")
train_data1 = np.load('/efs/users/zucksliu/env_project/wind_dataset/wind_tu_756_3_721_1440.npy')
train_data2 = np.load('/efs/users/zucksliu/env_project/wind_dataset/wind_tv_756_3_721_1440.npy')
p_min1 = [199.93072509765625, -19.626953125, -4934.843585180133]  # u10
max_min1 = [315.47837829589844 - 199.93072509765625, 16.193924903869625 + 19.626953125, 3888.185005817384 + 4934.843585180133]  # u10
p_min2 = [199.93072509765625, -16.156066894531254, -4381.68313198965] #v10
max_min2 = [315.47837829589844 - 199.93072509765625, 20.362215995788574 + 16.156066894531254, 4969.35201622079 + 4381.68313198965] #v10
train_transforms1 = transforms.Compose(
        [transforms.Normalize(p_min1, max_min1),]
)
train_transforms2 = transforms.Compose(
        [transforms.Normalize(p_min2, max_min2), ]
)
tr1 = torch.from_numpy(train_data1)
tr2 = torch.from_numpy(train_data2)
train_dataset1 = CustomTensorDataset(tr1, transforms = train_transforms1, dataset_index=0)
train_dataset2 = CustomTensorDataset(tr2, transforms = train_transforms2, dataset_index=1)
train_dataset = [train_dataset2,]
concat_dataset_train = ConcatDataset(train_dataset)
train_dataloader = DataLoader(concat_dataset_train, batch_size=24, num_workers=4,
                              shuffle=False, pin_memory=(device == device),)

Keys = ['temp_mse', 'salt_mse', 'zeta_mse', 'temp_mse_target','salt_mse_target','zeta_mse_target', 'mse_target_x_hat',   'psnr','psnr_target',  'msssim', 'msssim_target', 'bpp']
metrics = {}
for name, _ in networks.items():
    metrics[name] = dict.fromkeys(Keys, 0)

grid_location_embedding_padding = construct_location_embedding_padding(long_embedding, lati_embedding, 0, 0, lat_res=721, long_res=1440, long_dim=4, lat_dim=4, padding_direction=[16, 16, 23, 24])
print('grid_location_embedding_padding', grid_location_embedding_padding.shape)
# anchor_index = np.arange(0, 757, 2)
# target_index = np.arange(1, 756, 2)
data_number = 0
with torch.no_grad():
    for i, d in enumerate(train_dataloader):
        if i > 30:
            continue
        d, time_index, dataset_index = d
        data_number = i
        d = d.to(torch.float32)
        d = d.to(device)
        bs, c, h, w = d.size(0), d.size(1), d.size(2), d.size(3)
        max_min = torch.zeros(bs, c)
        p_min = torch.zeros(bs, c)
        for j in range(bs):
            for k in range(c):
                if dataset_index[j] == 0:
                    max_min[j, k] = max_min1[k]
                    p_min[j, k] = p_min1[k]
                else:
                    max_min[j, k] = max_min2[k]
                    p_min[j, k] = p_min2[k]

        for name, net in networks.items():
            p = 64 
             # maximum 6 strides of 2
            new_h = (h + p - 1) // p * p
            new_w = (w + p - 1) // p * p
            # new_h = h // p * p
            # new_w = w // p * p
            padding_left = (new_w - w) // 2
            padding_right = new_w - w - padding_left
            padding_top = (new_h - h) // 2
            padding_bottom = new_h - h - padding_top
            # print(padding_top, padding_bottom)
            # sleep
            x_padded = F.pad(d, (padding_left, padding_right, padding_top, padding_bottom),
                mode="constant", value=0)
            # x_padded = d[:, :, :new_h, :new_w]
            left_list, top_list = torch.zeros(bs, dtype=int), torch.zeros(bs, dtype=int)
            left_list = left_list + 16
            top_list = top_list + 23
            # loc_emb = batch_location_embedding(grid_location_embedding_padding, left=left_list, top=top_list,
                                            #    long_res=new_w, lat_res=new_h, loc_res=8)
            loc_emb = grid_location_embedding_padding
            loc_emb = loc_emb.expand(bs, loc_emb.size(1), loc_emb.size(2), loc_emb.size(3))
            loc_emb = torch.permute(loc_emb, (0, 3, 1, 2))
            # print(x_padded.shape, loc_emb.shape)
            # print(loc_emb.shape)
            # sleep

            # x_anchor = x_padded[time_index % 2 == 0]
            # target_x = x_padded[time_index % 2 == 1]
            # anchor_time_index = time_index[time_index % 2 == 0]
            # new_time_index = time_index[time_index % 2 == 1]

            # loc_emb_anchor = loc_emb[(time_index % bs) % 2 == 0]
            # loc_emb_target = loc_emb[(time_index % bs) % 2 == 1]

            x_anchor = x_padded[:12]
            target_x = x_padded[12:]
            anchor_time_index = time_index[:12]
            new_time_index = time_index[12:]

            loc_emb_anchor = loc_emb[:12]
            loc_emb_target = loc_emb[12:]



            # print(anchor_time_index, new_time_index)
            # sleep
            # target_x = fetch_target_data(concat_dataset_train, new_time_index, left_list, top_list, dataset_index=dataset_index, patch_size=patch_size)
            
            x_anchor_time_embedding = fetch_time_embedding(anchor_time_index, all_time_index_embedding)
            target_x_time_embedding = fetch_time_embedding(new_time_index, all_time_index_embedding)
            # print(x_anchor_time_embedding)
            # print(x_anchor.device, target_x.device, anchor_time_index.device, new_time_index.device, loc_emb.device, x_anchor_time_embedding.device, target_x_time_embedding.device)
            # print("device:",device)
            # device = next(net.parameters()).device
            loc_emb_anchor = loc_emb_anchor.to(device)
            loc_emb_target = loc_emb_target.to(device)
            x_anchor_time_embedding = x_anchor_time_embedding.to(device)
            target_x_time_embedding = target_x_time_embedding.to(device)
            # print(x_anchor.device, target_x.device, anchor_time_index.device, new_time_index.device, loc_emb.device, x_anchor_time_embedding.device, target_x_time_embedding.device)

            net.eval()
            # print(next(net.parameters()).device)
            # print(x_anchor.device, target_x.device, anchor_time_index.device, new_time_index.device, loc_emb.device, x_anchor_time_embedding.device, target_x_time_embedding.device)
            
            # sleep
            with torch.no_grad():
                rv = net(x_anchor, target_x, x_anchor_time_embedding, target_x_time_embedding, loc_emb_anchor, loc_emb_target, 2)
            target_max_min = max_min[time_index % 2 == 1]
            target_p_min = p_min[time_index % 2 == 1]
            target_max_min = target_max_min.to(device)
            target_p_min = target_p_min.to(device)
            # print(x_anchor.shape, target_x.shape)
            # print(rv['x_hat'].shape, rv['target_x_hat'].shape)
            # sleep
            rv['recover_x_hat'] = torch.stack([torch.add(torch.mul(rv['x_hat'][:, i, :, :], target_max_min[:, i].reshape(-1, 1, 1)), target_p_min[:, i].reshape(-1, 1, 1)) for i in
                 range(3)], dim=1)
            rv['recover_target_x_hat'] = torch.stack(
                [torch.add(torch.mul(rv['target_x_hat'][:, i, :, :], target_max_min[:, i].reshape(-1, 1, 1)), target_p_min[:, i].reshape(-1, 1, 1)) for i in
                 range(3)], dim=1)
            # print(rv['recover_x_hat'].shape, rv['recover_target_x_hat'].shape)
            # print(new_time_index, anchor_time_index, bs)
            ori_x = torch.stack(
                [torch.add(torch.mul(x_padded[new_time_index % bs, i, :, :], target_max_min[:, i].reshape(-1, 1, 1)), target_p_min[:, i].reshape(-1, 1, 1)) for i in
                 range(3)], dim=1)
            # print(ori_x[:, 0, :, :])
            # print(rv['recover_x_hat'][:, 0, :, :])
            # print(rv['recover_target_x_hat'][:, 0, :, :])            
            # print(ori_x[anchor_time_index % bs, 0, :, :])
            # print(rv['recover_x_hat'][:, 0, :, :])
            # print(rv['recover_target_x_hat'][:, 0, :, :])
            # sleep


            # left_index = 0
            # right_index = new_h
            # top_index = 0
            # bottom_index = new_w

            left_index = padding_left
            right_index = padding_left + h
            top_index = padding_top
            bottom_index = padding_top + w

            metrics[name]['temp_mse'] += mse(ori_x[:, 0, left_index:right_index, top_index:bottom_index],
                                             rv['recover_x_hat'][:, 0, left_index:right_index, top_index:bottom_index])
            metrics[name]['salt_mse'] += mse(ori_x[:, 1, left_index:right_index, top_index:bottom_index],
                                             rv['recover_x_hat'][:, 1, left_index:right_index, top_index:bottom_index])
            metrics[name]['zeta_mse'] += mse(ori_x[:, 2, left_index:right_index, top_index:bottom_index],
                                             rv['recover_x_hat'][:, 2, left_index:right_index, top_index:bottom_index])

            metrics[name]['temp_mse_target'] += mse(ori_x[:, 0, left_index:right_index, top_index:bottom_index],
                                                    rv['recover_target_x_hat'][:, 0, left_index:right_index, top_index:bottom_index])
            metrics[name]['salt_mse_target'] += mse(ori_x[:, 1, left_index:right_index, top_index:bottom_index],
                                                    rv['recover_target_x_hat'][:, 1, left_index:right_index, top_index:bottom_index])
            metrics[name]['zeta_mse_target'] += mse(ori_x[:, 2, left_index:right_index, top_index:bottom_index],
                                                    rv['recover_target_x_hat'][:, 2, left_index:right_index, top_index:bottom_index])

            metrics[name]['mse_target_x_hat'] += mse(rv['recover_target_x_hat'], rv['recover_x_hat'])
            metrics[name]['psnr'] += compute_psnr(ori_x[:, :, left_index:right_index, top_index:bottom_index],
                                                  rv['recover_x_hat'][:, :, left_index:right_index, top_index:bottom_index])
            metrics[name]['psnr_target'] += compute_psnr(ori_x[:, :, left_index:right_index, top_index:bottom_index],
                                                         rv['recover_target_x_hat'][:, :, left_index:right_index, top_index:bottom_index])
            metrics[name]['msssim'] += compute_msssim(ori_x[:, :, left_index:right_index, top_index:bottom_index],
                                                      rv['recover_x_hat'][:, :, left_index:right_index, top_index:bottom_index])
            metrics[name]['msssim_target'] += compute_msssim(ori_x[:, :, left_index:right_index, top_index:bottom_index],
                                                             rv['recover_target_x_hat'][:, :, left_index:right_index, top_index:bottom_index])
            metrics[name]['bpp'] += compute_bpp(rv)
            print(i)

print(data_number)
Var = [445.5796776485072, 15.041957076262898, 1176447.5448170535] #u10
for name in metrics.keys():
    metrics[name]['temp_mse'] = np.sqrt(metrics[name]['temp_mse'] / data_number / Var[0])
    metrics[name]['salt_mse'] = np.sqrt(metrics[name]['salt_mse'] / data_number / Var[1])
    metrics[name]['zeta_mse'] = np.sqrt(metrics[name]['zeta_mse'] / data_number / Var[2])
    metrics[name]['temp_mse_target'] = np.sqrt(metrics[name]['temp_mse_target']/ data_number/ Var[0])
    metrics[name]['salt_mse_target'] = np.sqrt(metrics[name]['salt_mse_target'] / data_number / Var[1])
    metrics[name]['zeta_mse_target'] = np.sqrt(metrics[name]['zeta_mse_target'] / data_number / Var[2])
    metrics[name]['mse_target_x_hat'] = metrics[name]['mse_target_x_hat'] / data_number
    metrics[name]['psnr'] = metrics[name]['psnr'] / data_number
    metrics[name]['psnr_target'] = metrics[name]['psnr_target'] / data_number
    metrics[name]['msssim'] = metrics[name]['msssim'] / data_number
    metrics[name]['msssim_target'] = metrics[name]['msssim_target'] / data_number
    metrics[name]['bpp'] = metrics[name]['bpp'] / data_number
print(metrics)
# reconstructions = {name: transforms.ToPILImage()(out['x_hat'].squeeze())
#                    for name, out in outputs.items()}

header = f'{"Model":30s} | {"t2m_NMSE":<9s} | {"u10_NMSE":<9s} |  {"t2m_u10_NMSE":<9s} | {"t2m_target_NMSE":<9s} |{"u10_target_NMSE":<9s}|{"t2m_u10_target_NMSE":<9s}|{"x_hat_target_MSE":<9s}|{"PSNR [dB]"} | {"MS-SSIM":<9s} | {"Bpp":<9s}'
print('-' * len(header))
print(header)
print('-' * len(header))
for name, m in metrics.items():
    print(f'{name:35s}', end='')
    for v in m.values():
        print(f' | {v:9.5f}', end='')
    print('|')
print('-' * len(header))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
for name, m in metrics.items():
    axes[0].plot(m['bpp'], m['psnr'], 'o', label=name)
    axes[0].legend(loc='best')
    axes[0].grid()
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Bit-rate [bpp]')
    axes[0].title.set_text('PSNR comparison')

    axes[1].plot(m['bpp'], -10 * np.log10(1 - m['msssim']), 'o', label=name)
    axes[1].legend(loc='best')
    axes[1].grid()
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Bit-rate [bpp]')
    axes[1].title.set_text('MS-SSIM (log) comparison')

plt.show()