import math
# import io
import torch
from torchvision import transforms
import numpy as np
#

# from collections import defaultdict
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
# from compressai.zoo import cheng2020_anchor
from compressai.utils.eval_model.__main__ import load_checkpoint, collect_images, read_image, eval_model

# p = 'frames299.yuv'
# # x = np.asarray(Image.open(p))
# f = open(p, 'rb')
# x = f.read()
# print(x)
# exit()
# x1 = np.load('/data/zixinl6/Compress/compressai2/g_a_756_192_16_16_1.npy')
# x2 = np.load('/data/zixinl6/Compress/compressai2/g_a_756_192_16_16_2.npy')
# x3 = np.load('/data/zixinl6/Compress/compressai2/h_a_756_192_4_4_1.npy')
# x4 = np.load('/data/zixinl6/Compress/compressai2/h_a_756_192_4_4_2.npy')
# y1 = np.concatenate([x1, x2], axis=0)
# np.save('cheng_ocean_wind_g_a_1512_192_16_16', y1)
# print(y1.shape)
# y2 = np.concatenate([x3, x4], axis=0)
# np.save('cheng_ocean_wind_h_a_1512_192_4_4', y2)
# print(y2.shape)
# exit()
# val1 = np.load("/data/zixinl6/downscaling/valid1_92_3_1302_663.npy")
# val1 = val1[0]
# print(val1.shape)
# print(np.amin(val1))
# print(np.where(val1 == -2.3634198))
# print(np.amax(np.where(val1==1.000e+20, -100, val1)))
# print(np.where(val1 == 32.947636))
# temp = val1[0]
# print(np.amin(temp))
# print(np.where(temp ==0.0))
# print(np.amax(np.where(temp == 1.0000000e+20, -100, temp)))
# print(np.where(temp == 12.187124))
# q = val1
# x = torch.from_numpy(q)
# # an_array = np.where(q == 1.0000e+20, 0, q)
# l1 = x.mul(255).byte()
# z = transforms.ToPILImage()(x)
#
# exit()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--channel', action="store_true", default =False, help="change the third channel to temp multiply salt")
parser.add_argument('-s', '--size', action="store_true", default = False, help="change the third channel to temp multiply salt")
args = vars(parser.parse_args())

def mse(a, b):
    return torch.mean((a - b)**2).item()

def compute_psnr(a, b):
    # print(a)
    # print(b)
    mse = torch.mean((a - b)**2).item()
    # print(mse)
    # exit()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()
# def compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#               for likelihoods in out_net['likelihoods'].values()).item()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
metric = 'mse'  # only pre-trained model for mse are available for now
quality = 6    # lower quality -> lower bit-rate (use lower quality to clearly see visual differences in the notebook)
# #
# epoch_100 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_100-9b35db0f.pth.tar')
# epoch_150 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_150-576b42f0.pth.tar')
# epoch_200 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_200-a233406b.pth.tar')
# epoch_250 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_250-eb9a63f1.pth.tar')
# epoch_300 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_300-6623ef6a.pth.tar')
# epoch_400 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_400-f706f9ad.pth.tar')
# epoch_450 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_450-c0791787.pth.tar')
# epoch_500 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_500-9853741f.pth.tar')
# epoch_550 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_550-d681541a.pth.tar')
# epoch_600 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_600-04bd4221.pth.tar')
# epoch_700 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_700-c3154f58.pth.tar')
# epoch_799 = load_checkpoint('bmshj2018-hyperprior', '/data/zixinl6/Compress/compressai2/bmshj2018_hyper_updated/bmshj2018-hyperprior_wind_epoch_799-2cc12bba.pth.tar')

epoch_100 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_100-ba64433a.pth.tar')
epoch_150 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_150-b801d0c3.pth.tar')
epoch_200 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_200-1bcc1834.pth.tar')
epoch_250 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_250-ad6d9834.pth.tar')
epoch_300 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_300-676d9631.pth.tar')
epoch_350 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_350-9fef7594.pth.tar')
epoch_400 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_400-c48a07f1.pth.tar')
epoch_500 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_500-1f9f3439.pth.tar')
epoch_550 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_550-28095711.pth.tar')
epoch_600 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_600-c808ec39.pth.tar')
epoch_700 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_700-f61e90b9.pth.tar')
epoch_799 = load_checkpoint('mbt2018-mean', '/data/zixinl6/Compress/compressai2/mbt2018_mean_updated/mbt2018-mean_wind_epoch_799-3959ae50.pth.tar')









# epoch_100 = load_checkpoint('cheng2020-anchor-transfer', '/data/zixinl6/Compress/compressai2/cheng2020-anchor-transfer_wind_epoch_100-6b359dc4.pth.tar')
# epoch_0 = load_checkpoint('cheng2020-anchor-transfer', '/data/zixinl6/Compress/compressai2/cheng2020-anchor-transfer_wind_epoch_0-470cb912.pth.tar')
# epoch_25 = load_checkpoint('cheng2020-anchor-transfer', '/data/zixinl6/Compress/compressai2/cheng2020-anchor-transfer_wind_epoch_25-5c04d4da.pth.tar')
# epoch_10 = load_checkpoint('cheng2020-anchor', "/data/zixinl6/Compress/compressai/checkpoint_10_updated_cheng2020.pth.tar-266c1d46.pth.tar")
# epoch_100 = load_checkpoint('cheng2020-anchor', "/data/zixinl6/Compress/compressai/checkpoint_100_updated_cheng2020.pth.tar")
# epoch_100_random = load_checkpoint('cheng2020-anchor', "/data/zixinl6/Compress/compressai/checkpoint_100_random_updated_cheng2020.pth.tar-b3767d9e.pth.tar")
# epoch_600 = load_checkpoint('cheng2020-anchor', "/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_600_updated.pth.tar-34ab5fd2.pth.tar")
# epoch_500 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_500_updated.pth.tar-c9120bbb.pth.tar')
# epoch_600 = load_checkpoint('cheng2020-anchor', "/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_600_updated.pth.tar-44b8070c.pth.tar")
# epoch_700 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_700_updated.pth.tar-7b6a0401.pth.tar')
#     epoch_5 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_5_updated_2.pth.tar-1579c308.pth.tar')
#     epoch_45 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_45_updated.pth.tar-81632e14.pth.tar')
#     epoch_75 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_75_updated.pth.tar-e3ea8acf.pth.tar')
#     epoch_100 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_100_updated.pth.tar-4f05720d.pth.tar')
#     epoch_125 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_125_updated.pth.tar-ea0c5aad.pth.tar')
#     epoch_150 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_150_updated.pth.tar-81484a68.pth.tar')
#     epoch_175 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_175_updated.pth.tar-62795b5e.pth.tar')
#     epoch_200 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_200_updated.pth.tar-a1acb9f5.pth.tar')
#     epoch_300 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_300_updated.pth.tar-b4f0b402.pth.tar')
#     epoch_400 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_400_updated.pth.tar-fcd9236c.pth.tar')
#     epoch_500 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_500_updated.pth.tar-dea53ce6.pth.tar')
#     epoch_600 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_600_updated.pth.tar-1f26f80e.pth.tar')
#     epoch_750 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_750_updated.pth.tar-3e4b3175.pth.tar')
#     epoch_799 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_799_updated.pth.tar-ebd1d5d6.pth.tar')
# epoch_399 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_epoch_399_updated.pth.tar-34ace686.pth.tar')
#     epoch_50 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality6_cheng2020_epoch_50_updated.pth.tar-7e0f494c.pth.tar')
#     epoch_75_2  = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality6_cheng2020_epoch_75_updated.pth.tar-a71856f0.pth.tar')
#     epoch_100_2 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality6_cheng2020_epoch_100_updated.pth.tar-f1c2ddec.pth.tar')
#     epoch_125_2 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality6_cheng2020_epoch_125_updated.pth.tar-82236aa3.pth.tar')
#     epoch_150_2 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality6_cheng2020_epoch_150_updated.pth.tar-28d2a747.pth.tar')
#     epoch_175_2 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality6_cheng2020_epoch_175_updated.pth.tar-5376c790.pth.tar')
#     epoch_200_2 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality6_cheng2020_epoch_200_updated.pth.tar-27e91ffe.pth.tar')
# best = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_cheng2020_best.pth.tar-347e6a79.pth.tar')
#     epoch_25_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_25_updated.pth.tar-d2e83735.pth.tar')
#     epoch_50_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_50_updated.pth.tar-c776ba93.pth.tar')
#     epoch_75_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_75_updated.pth.tar-f97f379d.pth.tar')
#     epoch_100_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_100_updated.pth.tar-5ba9a3c8.pth.tar')
#     epoch_125_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_125_updated.pth.tar-1ddad9e0.pth.tar')
#     epoch_150_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_150_updated.pth.tar-8c3a6065.pth.tar')
#     epoch_175_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_175_updated.pth.tar-16036be7.pth.tar')
#     epoch_200_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_200_updated.pth.tar-665bda64.pth.tar')
#     epoch_225_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_225_updated.pth.tar-3c8b7d7d.pth.tar')
#     epoch_300_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_300_updated.pth.tar-47c5aa63.pth.tar')
#     epoch_350_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_350_updated.pth.tar-604a632e.pth.tar')
#     epoch_375_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_375_updated.pth.tar-97ffcca7.pth.tar')
#     epoch_400_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_400_updated.pth.tar-90ddc755.pth.tar')
#     epoch_450_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_450_updated.pth.tar-272baf06.pth.tar')
#     epoch_475_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_475_updated.pth.tar-261cd785.pth.tar')
#     epoch_500_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_500_updated.pth.tar-6b70804c.pth.tar')
#     epoch_525_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_525_updated.pth.tar-3d575f6b.pth.tar')
#     epoch_550_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_550_updated.pth.tar-195fd597.pth.tar')
#     epoch_575_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_575_updated.pth.tar-4df0d8bd.pth.tar')
#     epoch_600_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_600_updated.pth.tar-4dafb5e0.pth.tar')
#     epoch_700_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_700_updated.pth.tar-8b3ecb0b.pth.tar')
#     epoch_799_3 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai/ocean_quality3_cheng2020_epoch_799_updated.pth.tar-23dd7551.pth.tar')
#     epoch_50_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_50_updated.pth.tar-18b37d82.pth.tar')
#     epoch_75_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_75_updated.pth.tar-d3121f23.pth.tar')
#     epoch_100_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_100_updated.pth.tar-41ca1630.pth.tar')
#     epoch_125_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_125_updated.pth.tar-a7bace44.pth.tar')
#     epoch_150_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_150_updated.pth.tar-d32d938d.pth.tar')
#     epoch_175_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_175_updated.pth.tar-ffd91fec.pth.tar')
#     epoch_225_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_225_updated.pth.tar-56c1ae49.pth.tar')
#     epoch_325_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_325_updated.pth.tar-a3fd1731.pth.tar')
#     epoch_425_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_425_updated.pth.tar-50bd8998.pth.tar')
#     epoch_525_4 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_cheng2020_epoch_525_updated.pth.tar-8dfb7b9c.pth.tar')
#     epoch_50_5 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_50_updated_1.pth.tar-221b8dc8.pth.tar')

# epoch_50_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_50_updated_2.pth.tar-33be19be.pth.tar')
# epoch_75_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_75_updated_2.pth.tar-7d4bef48.pth.tar')
# epoch_100_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_100_updated_2.pth.tar-0da9a1dd.pth.tar')
# epoch_125_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_125_updated_2.pth.tar-764e87cb.pth.tar')
# epoch_150_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_150_updated_2.pth.tar-13c70af1.pth.tar')
# epoch_175_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_175_updated_2.pth.tar-39206963.pth.tar')
# epoch_200_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_200_updated_2.pth.tar-17b5a559.pth.tar')
# epoch_225_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_225_updated_2.pth.tar-29c088cd.pth.tar')
# epoch_250_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_250_updated_2.pth.tar-7a560f0e.pth.tar')
# epoch_275_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_275_updated_2.pth.tar-251577cf.pth.tar')
# epoch_325_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_325_updated_2.pth.tar-9cd8dbe5.pth.tar')
# epoch_350_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_350_updated_2.pth.tar-ae14a071.pth.tar')
# epoch_375_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_375_updated_2.pth.tar-8332e141.pth.tar')
# epoch_400_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_400_updated_2.pth.tar-ea71a9d3.pth.tar')
# epoch_425_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_425_updated_2.pth.tar-b6d52b5a.pth.tar')
# epoch_450_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_450_updated_2.pth.tar-5602fe41.pth.tar')
# epoch_475_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_475_updated_2.pth.tar-dbfb1eb1.pth.tar')
# epoch_500_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_500_updated_2.pth.tar-3192f7bb.pth.tar')
# epoch_525_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_525_updated_2.pth.tar-7a51903a.pth.tar')
# epoch_550_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_550_updated_2.pth.tar-bc59a563.pth.tar')
# epoch_575_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_575_updated_2.pth.tar-e110e59d.pth.tar')
# epoch_600_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_600_updated_2.pth.tar-e2e43270.pth.tar')
# epoch_650_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_650_updated_2.pth.tar-9eb270d9.pth.tar')
# epoch_700_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch700_updated_2.pth.tar-39b967e5.pth.tar')
# epoch_750_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch750_updated_2.pth.tar-020b6e4a.pth.tar')
# epoch_799_6 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch799_updated_2.pth.tar-8dec2980.pth.tar')

# epoch_800_7 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_800_updated_2.pth.tar-7800ae65.pth.tar')
# epoch_850_7 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_850_updated_2.pth.tar-23b0e506.pth.tar')
# epoch_900_7 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_900_updated_2.pth.tar-d8b755b6.pth.tar')
# epoch_950_7 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_950_updated_2.pth.tar-8aac97e2.pth.tar')
# epoch_999_7 = load_checkpoint('cheng2020-anchor', '/data/zixinl6/Compress/compressai2/ocean0_quality6_lambda0.8_cheng2020_epoch_999_updated_2.pth.tar-dc5b84f8.pth.tar')

networks = {
    # 'mbt2018_epoch100' : epoch_100.to(device),
    # 'mbt2018_epoch150' : epoch_150.to(device),
    # 'mbt2018_epoch200' : epoch_200.to(device),
    # 'mbt2018_epoch250' : epoch_250.to(device),
    # 'mbt2018_epoch300' : epoch_300.to(device),
    # 'mbt2018_epoch400' : epoch_400.to(device),
    # 'mbt2018_epoch450' : epoch_450.to(device),
    # 'mbt2018_epoch500' : epoch_500.to(device),
    # 'mbt2018_epoch550' : epoch_550.to(device),
    # 'mbt2018_epoch600' : epoch_600.to(device),
    # 'mbt2018_epoch700' : epoch_700.to(device),
    # 'mbt2018_epoch799' : epoch_799.to(device),
    'balle2018_epoch100' : epoch_100.to(device),
    'balle2018_epoch150' : epoch_150.to(device),
    'balle2018_epoch200' : epoch_200.to(device),
    'balle2018_epoch250' : epoch_250.to(device),
    'balle2018_epoch300' : epoch_300.to(device),
    'balle2018_epoch350' : epoch_350.to(device),
    'balle2018_epoch400' : epoch_400.to(device),
    'balle2018_epoch500' : epoch_500.to(device),
    'balle2018_epoch550' : epoch_550.to(device),
    'balle2018_epoch600' : epoch_600.to(device),
    'balle2018_epoch700' : epoch_700.to(device),
    'balle2018_epoch799' : epoch_799.to(device),


    # 'epoch_25': epoch_25.to(device),
    # 'epoch_800_7': epoch_800_7.to(device),
    # 'epoch_850_7':epoch_850_7.to(device),
    # 'epoch_900_7':epoch_900_7.to(device),
    # 'epoch_950_7':epoch_950_7.to(device),
    # 'epoch_999_7':epoch_999_7.to(device),

    # 'cheng2020-anchor': cheng2020_anchor(quality=quality, pretrained=True).eval().to(device),
    # 'epoch50_lambda0.8_quality6': epoch_50_5.to(device),
    # 'epoch50_lambda0.8_quality6_2': epoch_50_6.to(device),
    # 'epoch75_lambda0.8_quality6_2': epoch_75_6.to(device),
    # 'epoch100_lambda0.8_quality6_2': epoch_100_6.to(device),
    # 'epoch125_lambda0.8_quality6_2': epoch_125_6.to(device),
    # 'epoch150_lambda0.8_quality6_2': epoch_150_6.to(device),
    # 'epoch175_lambda0.8_quality6_2': epoch_175_6.to(device),
    # 'epoch200_lambda0.8_quality6_2': epoch_200_6.to(device),
    # 'epoch225_lambda0.8_quality6_2': epoch_225_6.to(device),
    # 'epoch250_lambda0.8_quality6_2': epoch_250_6.to(device),
    # 'epoch275_lambda0.8_quality6_2': epoch_275_6.to(device),
    # # 'epoch325_lambda0.8_quality6_2': epoch_325_6.to(device),
    # 'epoch350_lambda0.8_quality6_2': epoch_350_6.to(device),
    # # 'epoch375_lambda0.8_quality6_2': epoch_375_6.to(device),
    # 'epoch400_lambda0.8_quality6_2': epoch_400_6.to(device),
    # # 'epoch425_lambda0.8_quality6_2': epoch_425_6.to(device),
    # 'epoch450_lambda0.8_quality6_2': epoch_450_6.to(device),
    # # 'epoch475_lambda0.8_quality6_2': epoch_475_6.to(device),
    # 'epoch500_lambda0.8_quality6_2': epoch_500_6.to(device),
    # # 'epoch525_lambda0.8_quality6_2': epoch_525_6.to(device),
    # 'epoch550_lambda0.8_quality6_2': epoch_550_6.to(device),
    # # 'epoch575_lambda0.8_quality6_2': epoch_575_6.to(device),
    # 'epoch600_lambda0.8_quality6_2': epoch_600_6.to(device),
    # 'epoch650_lambda0.8_quality6_2': epoch_650_6.to(device),
    # 'epoch700_lambda0.8_quality6_2': epoch_700_6.to(device),
    # 'epoch750_lambda0.8_quality6_2': epoch_750_6.to(device),
    # 'epoch799_lambda0.8_quality6_2': epoch_799_6.to(device),
    # 'epoch50_lambda10_quality6': epoch_50_4.to(device),
    # 'epoch75_lambda10_quality6': epoch_75_4.to(device),
    # 'epoch100_lambda10_quality6': epoch_100_4.to(device),
    # 'epoch125_lambda10_quality6': epoch_125_4.to(device),
    # 'epoch150_lambda10_quality6': epoch_150_4.to(device),
    # 'epoch175_lambda10_quality6': epoch_175_4.to(device),
    # 'epoch225_lambda10_quality6': epoch_225_4.to(device),
    # 'epoch325_lambda10_quality6': epoch_325_4.to(device),
    #
    # 'epoch425_lambda10_quality6': epoch_425_4.to(device),
    #
    # 'epoch525_lambda10_quality6': epoch_525_4.to(device),

    # 'epoch25_lambda0.8': epoch_25_3.to(device),
    # 'epoch50_lambda0.8': epoch_50_3.to(device),
    # 'epoch75_lambda0.8': epoch_75_3.to(device),
    # 'epoch100_lambda0.8': epoch_100_3.to(device),
    # 'epoch125_lambda0.8': epoch_125_3.to(device),
    # 'epoch150_lambda0.8': epoch_150_3.to(device),
    # 'epoch175_lambda0.8': epoch_175_3.to(device),
    # 'epoch200_lambda0.8': epoch_200_3.to(device),
    # 'epoch225_lambda0.8': epoch_225_3.to(device),
    # 'epoch300_lambda0.8': epoch_300_3.to(device),
    # 'epoch350_lambda0.8': epoch_350_3.to(device),
    # 'epoch375_lambda0.8': epoch_375_3.to(device),
    # 'epoch400_lambda0.8': epoch_400_3.to(device),
    # 'epoch450_lambda0.8': epoch_450_3.to(device),
    # 'epoch475_lambda0.8': epoch_475_3.to(device),
    # 'epoch500_lambda0.8': epoch_500_3.to(device),
    # 'epoch525_lambda0.8': epoch_525_3.to(device),
    # 'epoch550_lambda0.8': epoch_550_3.to(device),
    # 'epoch575_lambda0.8': epoch_575_3.to(device),
    # 'epoch600_lambda0.8': epoch_600_3.to(device),
    # 'epoch700_lambda0.8': epoch_700_3.to(device),
    # 'epoch799_lambda0.8': epoch_799_3.to(device),
    # 'epoch50_lambda0.1': epoch_50.to(device),
    # 'epoch75_lambda0.1': epoch_75_2.to(device),
    # 'epoch100_lambda0.1': epoch_100_2.to(device),
    # 'epoch125_lambda0.1': epoch_125_2.to(device),
    # 'epoch150_lambda0.1': epoch_150_2.to(device),
    # 'epoch175_lambda0.1': epoch_175_2.to(device),
    # 'epoch200_lambda0.1': epoch_200_2.to(device),
    #
    # 'epoch45_lambda0.01': epoch_45.to(device),

    # 'epoch75_lambda0.01': epoch_75.to(device),
    # 'epoch100_lambda0.01': epoch_100.to(device),
    # 'epoch125_lambda0.01': epoch_125.to(device),
    # 'epoch150_lambda0.01': epoch_150.to(device),
    # 'epoch175_lambda0.01': epoch_175.to(device),
    # 'epoch200_lambda0.01': epoch_200.to(device),
    # 'epoch300_lambda0.01': epoch_300.to(device),
    # 'epoch400_lambda0.01': epoch_400.to(device),
    # 'epoch500_lambda0.01': epoch_500.to(device),
    # 'epoch600_lambda0.01': epoch_600.to(device),
    # 'epoch750_lambda0.01': epoch_750.to(device),
    # 'epoch799_lambda0.01': epoch_799.to(device)
    # # 'cheng2020-anchor-best': best.to(device)
    # # 'cheng2020-anchor-epoch100-random': epoch_100_random.to(device),
}
# img = Image.oimg = Image.open('/data/zixinl6/Compress/CLIC_pro/test/alejandro-escamilla-6.png').convert('RGB')
# x = transforms.ToTensor()(img).unsqueeze(0).to(device)
# print(x.shape)
outputs = {}
# test = np.load("/data/zixinl6/downscaling/test_876_3_1302_663.npy")
# val1 = np.load("/data/zixinl6/downscaling/valid1_92_3_1302_663.npy")
# temp = np.load("/data/zixinl6/downscaling/t2m_756_721_1440.npy")
# v10 = np.load("/data/zixinl6/downscaling/v10_756_721_1440.npy")
# u10 = np.load("/data/zixinl6/downscaling/u10_756_721_1440.npy")
# z = np.zeros((756, 721, 1440))
# print(z.shape)
# val1 = np.stack([temp, u10, z], axis=1)
# val1.astype(np.float32)
# print(val1.shape)
# test_u10 = np.load("/data/zixinl6/downscaling/wind_test_tu_75_3_721_1440.npy")
test_v10 = np.load("/data/zixinl6/downscaling/wind_test_tv_75_3_721_1440.npy")
val1 = test_v10

# exit()
# val1 = np.load("/data/zixinl6/downscaling/train_7693_3_1302_663.npy")
# val1
# x = val1
# transforms = transforms.Compose(
#     [transforms.ToPILImage(), transforms.ToTensor()]
# )
# x = transforms(x)
# img  = val1[0, :, :, :]
# x = transforms.ToTensor()(img).to(device)
# print(x.shape)
# x = torch.permute(x, (1, 0, 2))
# img = torch.permute(x, (1, 2, 0))
# print(x.shape)
# plt.figure(figsize=(9, 6))
# plt.axis('off')
# plt.imshow(img)
# plt.show()

# filepaths = collect_images("/data/zixinl6/Compress/CLIC_pro/test")
# filepaths = collect_images("/data/zixinl6/Compress/Kodak")
# print(networks)
# tr[:, 2, :, :] = torch.mul(tr[:, 0, :, :], tr[:, 1, :, :])
# te[:, 2, :, :] = torch.mul(te[:, 0, :, :], te[:, 1, :, :])
Keys = ['temp_mse', 'salt_mse', 'zeta_mse', 'psnr', 'msssim', 'bpp']
metrics = {}
for name, _ in networks.items():
    metrics[name] = dict.fromkeys(Keys, 0)
# print(metrics)
# exit()
# if args['channel']:
#     p_min = [-0.81310266, -0.0005704858, -26.1313]
#     max_min = [28.265799 + 0.81310266, 35.88579 + 0.0005704858, 825.0345 + 26.1313]
# else:
#     p_min = [-0.81310266, -0.0005704858, -3.279859]
#     max_min = [28.265799 + 0.81310266, 35.88579 + 0.0005704858, 15.006741 + 3.279859]
p_min = [199.93072509765625, -16.156066894531254, -4381.68313198965] #v10
max_min = [315.47837829589844 - 199.93072509765625, 20.362215995788574 + 16.156066894531254, 4969.35201622079 + 4381.68313198965] #v10
Var = [445.5796776485072, 6.192518140223957, 451763.88486925024] #v10
# p_min = [199.93072509765625, -19.626953125, -4934.843585180133]  # u10
# max_min = [315.47837829589844 - 199.93072509765625, 16.193924903869625 +19.626953125,  3888.185005817384+4934.843585180133] #u10
# Var = [445.5796776485072, 15.041957076262898, 1176447.5448170535] #u10
# Var = [8.835172, 12.286177, 10248.476]
# t2m = test_v10[:, 2, :, :]
# print(t2m.shape)
# zeros = np.zeros((75, 721, 1440))
# val1 = np.stack((zeros, zeros, t2m), axis = 1)
# print(val1.shape)
# val1 = test_u10
# Var = [1,1,1]
data_number = val1.shape[0]
with torch.no_grad():
    for i in range(data_number):

        ori_x = val1[i, :, :, :].squeeze()
        ori_x = torch.from_numpy(ori_x)
        # x = ori_x.to(torch.float16)
        # print(x.dtype)
        x = ori_x
        # print(ori_x)
        # x[x == 1.0000e+20] = 100
        # if args['channel']:
        #     x[2, :, :] = torch.mul(x[0, :, :], x[1, :, :])
        # print(ori_x)
        # print(x)
        x = transforms.Normalize(p_min, max_min)(x)
        # print(x)
        # print(ori_x)
        # exit()
        # if args['size']:
        #     x = transforms.Resize((256, 256))(x)
        # print(x)
        # exit()
        # x = torch.nan_to_num(x, posinf=-1)
        # x = x.float()
        # print(x.dtype)
        # x = transforms(x)
        # print(x.shape)
        x = x.unsqueeze(0).to(device)
        x = x.to(torch.float32)
        # print(x.dtype)
        # exit()
        for name, net in networks.items():
            net.eval()
    #         e_metrics = eval_model(net, filepaths)
    #         metrics[name] = e_metrics
    #         print(net)
            h, w = x.size(2), x.size(3)
            p = 64  # maximum 6 strides of 2
            new_h = (h + p - 1) // p * p
            new_w = (w + p - 1) // p * p
            padding_left = (new_w - w) // 2
            padding_right = new_w - w - padding_left
            padding_top = (new_h - h) // 2
            padding_bottom = new_h - h - padding_top
            x_padded = F.pad(
                x,
                (padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            # x_padded = x
            # print(x_padded[0, 0, 21:1323, 20:683])
            # print(x_padded[0, 0, :, :])
            # print(x_padded.shape)

            # if args['channel']:
            #     input_x = x_padded + 0.1
            #     mask = torch.where(input_x > 1.1, 1, 0)
            #     input_x[input_x > 1.1] = 0
            # else:
            input_x = x_padded
            #     mask = torch.where(x_padded>1, 1, 0)
            # print(input_x[0, 0, 23:744, 16:1456])
            # print(input_x)
            # print(mask)
            # count = torch.count_nonzero(mask)
            # print(count)
            # print(mask.shape)
            # exit()
            rv = net(input_x)
            # print(rv['x_hat'])
            # print(rv['x_hat'][0, 0, :, :])
            # print(rv['x_hat'][0, 0, 23:744, 16:1456])
            # exit()
            # if args['channel']:
            #     rv['x_hat'] = rv['x_hat'] - 0.1
            # rv['x_hat'][mask == 1]=0


            # print(rv['x_hat'][0, 0, 21:1323, 20:683])
            # exit()
            # print(rv['x_hat'][0, 0, 21:1323, 20:683])
            # exit()
            # print(rv['x_hat'].shape)
            ori_padded = ori_x.to(device)
            # if args['size']:
            ori_padded = F.pad(
                ori_padded,
                (padding_left, padding_right, padding_top, padding_bottom),
                mode="constant",
                value=0,
            )
            # print(ori_padded.shape)

            # ori_padded = transforms.Resize((256, 256))(ori_padded)
            # rv['recover_x_hat_temp'] = rv['x_hat'][0, 0, :, :].squeeze()
            # rv['recover_x_hat_temp'] = torch.add(torch.mul(rv['recover_x_hat_temp'], max_min[0]), p_min[0])[21:1323, 20:683]
            # rv['recover_x_hat_salt'] = rv['x_hat'][0, 1, :, :].squeeze()
            # rv['recover_x_hat_salt'] = torch.add(torch.mul(rv['recover_x_hat_salt'], max_min[1]), p_min[1])[21:1323, 20:683]
            # rv['recover_x_hat_zeta'] = rv['x_hat'][0, 2, :, :].squeeze()
            # rv['recover_x_hat_zeta'] = torch.add(torch.mul(rv['recover_x_hat_zeta'], max_min[2]), p_min[2])[21:1323, 20:683]

            rv['recover_x_hat_temp'] = rv['x_hat'][0, 0, :, :].squeeze()
            rv['recover_x_hat_temp'] = torch.add(torch.mul(rv['recover_x_hat_temp'], max_min[0]), p_min[0])
            rv['recover_x_hat_salt'] = rv['x_hat'][0, 1, :, :].squeeze()
            rv['recover_x_hat_salt'] = torch.add(torch.mul(rv['recover_x_hat_salt'], max_min[1]), p_min[1])
            rv['recover_x_hat_zeta'] = rv['x_hat'][0, 2, :, :].squeeze()
            rv['recover_x_hat_zeta'] = torch.add(torch.mul(rv['recover_x_hat_zeta'], max_min[2]), p_min[2])

            # t = torch.where(ori_padded[0, :, :] == 1.0000e+20, 1.0000e+20, rv['recover_x_hat_temp'].double())
            # rv['recover_x_hat_temp'] = t.float()
            #
            # s = torch.where(ori_padded[1, :, :] == 1.0000e+20, 1.0000e+20, rv['recover_x_hat_salt'].double())
            # rv['recover_x_hat_salt'] = s.float()
            #
            # z = torch.where(ori_padded[2, :, :] == 1.0000e+20, 1.0000e+20, rv['recover_x_hat_zeta'].double())
            # # rv['recover_x_hat_zeta'] = z.float()
            # print(rv['recover_x_hat_temp'].shape)
            # print(ori_padded.shape)
            # rv['recover_x_hat_temp'][ori_padded[0, :, :] == 100]= 100
            # rv['recover_x_hat_salt'][ori_padded[1, :, :] == 100]= 100
            # rv['recover_x_hat_zeta'][ori_padded[2, :, :] == 100]= 100
            # rv['recover_x_hat_temp'][mask[0, 0, :, :] == 1] = 0
            # rv['recover_x_hat_salt'][mask[0, 1, :, :] == 1] = 0
            # rv['recover_x_hat_zeta'][mask[0, 2, :, :] == 1] = 0
            # print(rv['recover_x_hat_temp'])
            # print(torch.count_nonzero(mask))
            # print(mask.shape)
            # # print(ori_padded[2, :, :])
            # exit()
            rv['recover_x_hat'] = torch.stack((rv['recover_x_hat_temp'], rv['recover_x_hat_salt'], rv['recover_x_hat_zeta']), dim = 0).unsqueeze(0)
            # print(rv['recover_x_hat'].shape)
            # print(rv['recover_x_hat'][:, :, 21:1323, 20:683].shape)
            # print(rv['recover_x_hat'][:, :, 21:1323, 20:683])
            # exit()
            # print(rv['recover_x_hat_temp'])
            # print(rv['recover_x_hat_temp'].shape)
            # print(y)
            # exit()
            # ori_padded[mask[0, :, :, :] == 1] = 0
            # print(torch.mul(ori_padded[0, :, :], ori_padded[1, :, :]))
            # t_s = torch.nan_to_num(torch.mul(ori_padded[0, :, :], ori_padded[1, :, :]), posinf=0)
            # t_s_hat = torch.nan_to_num(torch.mul(rv['recover_x_hat_temp'], rv['recover_x_hat_salt']), posinf=0)
            # print("here")
            # print(rv['recover_x_hat_temp'])

            # torch.mean((a - b) ** 2).item()
            # print(torch.mean((rv['recover_x_hat_temp'] - ori_padded[0, :, :])**2))
            # check = (rv['recover_x_hat_temp'] - ori_padded[0, :, :]).cpu().numpy()
            # plt.hist(check)
            # plt.show()
            # exit()
            # print(input_x)
            # print(x_padded)
            # print(x_padded.shape)
            # print(ori_padded[0, :, :][21:1323, 20:683])
            # print(rv['recover_x_hat_temp'])
            #
            # print(ori_padded.shape)
            # print(rv['recover_x_hat_temp'][23:744, 16:1456].shape)
            # print(rv['recover_x_hat_temp'][21:1323, 20:683])

            # pp = ori_padded[2, :, :][21:1323, 20:683] - rv['recover_x_hat_zeta'][21:1323, 20:683]
            # # print(pp.shape)
            # # print(torch.max(torch.abs(pp)))
            # z = torch.max(torch.abs(pp)).item()
            # plt.figure()
            # f, axarr = plt.subplots(1, 2)
            # vmin = torch.min(ori_padded[2, :, :][21:1323, 20:683]).item()
            # vmax = torch.max(ori_padded[2, :, :][21:1323, 20:683]).item()
            # axarr[0].imshow(ori_padded[2, :, :][21:1323, 20:683].cpu().numpy(), vmin = vmin, vmax = vmax)
            # axarr[1].imshow(rv['recover_x_hat_zeta'][21:1323, 20:683].cpu().numpy(), vmin = vmin, vmax = vmax)
            #
            # f.suptitle('Live ocean temp * salt')
            # axarr[0].set_title('original')
            # axarr[1].set_title('recover')

            # plt.savefig('t_live_o_r.png')
            # plt.imshow(ori_padded[1, :, :][23:744, 16:1456].cpu().numpy(), vmin =-17, vmax = 21)
            # print(ori_padded[1, :, :][23:744, 16:1456])
            # plt.imshow(rv['recover_x_hat_salt'][23:744, 16:1456].cpu().numpy(), vmin =-17, vmax = 21)
            # plt.imshow(abs(pp.cpu().numpy()), vmax = 2)
            # # plt.matshow(abs(pp.cpu().numpy()))
            # plt.colorbar()
            # plt.title("temp * salt difference, max diff abs: "+"{:.2f}".format(torch.max(torch.abs(pp)).item()))
            # plt.show()
            # exit()
            # d4_tensor = ori_padded.unsqueeze(0)
            ori_padded = ori_padded.to(torch.float32)
            # print(ori_padded[0, :, :][23:744, 16:1456].shape)
            # print(ori_padded[0, :, :][23:-24, 16:-16].shape)
            # print(ori_padded[0, :, :][23:-24, 16:-16])
            # exit()
            # print(rv['recover_x_hat_temp'].shape)
            # print(rv['recover_x_hat_temp'][23:744, 16:1456])

            # print(ori_padded[:, 21:1323, 20:683])
            # exit()
            # if name not in metrics.keys():
            #     # print(ori_padded[0, :, :][21:1323, 20:683].shape)
            #     metrics[name] = {
            #         # 'temp_mse': mse(ori_padded[0, :, :][43:-44, 36:-36], rv['recover_x_hat_temp'][43:-44, 36:-36]),
            #         # 'salt_mse': mse(ori_padded[1, :, :][43:-44, 36:-36], rv['recover_x_hat_salt'][43:-44, 36:-36]),
            #         # 'zeta_mse': mse(ori_padded[2, :, :][43:-44, 36:-36], rv['recover_x_hat_zeta'][43:-44, 36:-36]),
            #         'temp_mse': mse(ori_padded[0, :, :][23:744, 16:1456], rv['recover_x_hat_temp'][23:744, 16:1456]),
            #         'salt_mse': mse(ori_padded[1, :, :][23:744, 16:1456], rv['recover_x_hat_salt'][23:744, 16:1456]),
            #         'zeta_mse': mse(ori_padded[2, :, :][23:744, 16:1456], rv['recover_x_hat_zeta'][23:744, 16:1456]),
            #         # 'temp_mse': mse(ori_padded[0, :, :][21:1323, 20:683], rv['recover_x_hat_temp'][21:1323, 20:683]),
            #         # 'salt_mse': mse(ori_padded[1, :, :][21:1323, 20:683], rv['recover_x_hat_salt'][21:1323, 20:683]),
            #         # 'zeta_mse': mse(ori_padded[2, :, :][21:1323, 20:683], rv['recover_x_hat_zeta'][21:1323, 20:683]),
            #         # 'temp_salt_mse': mse(t_s, t_s_hat),
            #         # 't_mse': mse(x_padded[0, 0, :, :], rv['x_hat'][0, 0, :, :]),
            #         # 's_mse': mse(x_padded[0, 1, :, :], rv['x_hat'][0, 1, :, :]),
            #         # 'mse':  mse(x_padded, rv['x_hat']),
            #
            #         # 'psnr': compute_psnr(ori_padded[:, 21:1323, 20:683].unsqueeze(0), rv['recover_x_hat'][:, :, 21:1323, 20:683]),
            #         # 'msssim': compute_msssim(ori_padded[:, 21:1323, 20:683].unsqueeze(0), rv['recover_x_hat'][:, :, 21:1323, 20:683]),
            #         'psnr': compute_psnr(ori_padded[:, 23:744, 16:1456].unsqueeze(0),
            #                              rv['recover_x_hat'][:, :, 23:744, 16:1456]),
            #         'msssim': compute_msssim(ori_padded[:, 23:744, 16:1456].unsqueeze(0),
            #                                  rv['recover_x_hat'][:, :, 23:744, 16:1456]),
            #         'bpp': compute_bpp(rv)
            #     }
            #     # print(metrics)
            #     # exit()
            # else:
            metrics[name]['temp_mse'] += mse(ori_padded[0, :, :][23:744, 16:1456], rv['recover_x_hat_temp'][23:744, 16:1456])
            metrics[name]['salt_mse'] += mse(ori_padded[1, :, :][23:744, 16:1456], rv['recover_x_hat_salt'][23:744, 16:1456])
            metrics[name]['zeta_mse'] += mse(ori_padded[2, :, :][23:744, 16:1456], rv['recover_x_hat_zeta'][23:744, 16:1456])
            # metrics[name]['temp_mse'] += mse(ori_padded[0, :, :][21:1323, 20:683], rv['recover_x_hat_temp'][21:1323, 20:683])
            # metrics[name]['salt_mse'] += mse(ori_padded[1, :, :][21:1323, 20:683], rv['recover_x_hat_salt'][21:1323, 20:683])
            # metrics[name]['zeta_mse'] += mse(ori_padded[2, :, :][21:1323, 20:683], rv['recover_x_hat_zeta'][21:1323, 20:683])
            # metrics[name]['temp_salt_mse'] += mse(t_s, t_s_hat)
            # metrics[name]['t_mse'] += mse(x_padded[0, 0, :, :], rv['x_hat'][0, 0, :, :])
            # metrics[name]['s_mse'] += mse(x_padded[0, 1, :, :], rv['x_hat'][0, 1, :, :])
            # metrics[name]['mse'] += mse(x_padded, rv['x_hat'])
            metrics[name]['psnr'] += compute_psnr(ori_padded[:, 23:744, 16:1456].unsqueeze(0),
                                     rv['recover_x_hat'][:, :, 23:744, 16:1456])
            metrics[name]['msssim'] += compute_msssim(ori_padded[:, 23:744, 16:1456].unsqueeze(0),
                                         rv['recover_x_hat'][:, :, 23:744, 16:1456])
            # metrics[name]['psnr'] += compute_psnr(ori_padded[:, 21:1323, 20:683].unsqueeze(0), rv['recover_x_hat'][:, :, 21:1323, 20:683])
            # metrics[name]['msssim'] += compute_msssim(ori_padded[:, 21:1323, 20:683].unsqueeze(0), rv['recover_x_hat'][:, :, 21:1323, 20:683])
            metrics[name]['bpp'] += compute_bpp(rv)

            # print(ori_padded[0, :, :])
            # print(ori_padded.shape)
            # print(rv['recover_x_hat_temp'])
            # print(rv['recover_x_hat_temp'].shape)
            # print(x_padded.shape)
            # print(x_padded[0, 0, :, :])
            # print(rv['x_hat'][0, 0, :, :])
            # print(rv['x_hat'].shape)
            # exit()
            # print(metrics[name][])

print(data_number)
for name in metrics.keys():
    metrics[name]['temp_mse'] = np.sqrt(metrics[name]['temp_mse'] / data_number / Var[0])
    metrics[name]['salt_mse'] = np.sqrt(metrics[name]['salt_mse'] / data_number / Var[1])
    metrics[name]['zeta_mse'] = np.sqrt(metrics[name]['zeta_mse'] / data_number / Var[2])
    # metrics[name]['temp_salt_mse'] = metrics[name]['temp_salt_mse'] / data_number

    # metrics[name]['t_mse'] = metrics[name]['t_mse'] / data_number
    # metrics[name]['s_mse'] = metrics[name]['s_mse'] / data_number
    # metrics[name]['mse'] = metrics[name]['mse'] / data_number

    metrics[name]['psnr'] = metrics[name]['psnr'] /data_number
    metrics[name]['msssim'] = metrics[name]['msssim'] / data_number
    metrics[name]['bpp'] = metrics[name]['bpp']/data_number
print(metrics)
# outputs[name] = rv
reconstructions = {name: transforms.ToPILImage()(out['x_hat'].squeeze())
                  for name, out in outputs.items()}
# diffs = [torch.mean((out['x_hat'] - x).abs(), axis=1).squeeze()
#         for out in outputs.values()]
# exit()
# fix, axes = plt.subplots((len(reconstructions) + 2) // 3, 3, figsize=(16, 12))
# for ax in axes.ravel():
#     ax.axis('off')
#
# axes.ravel()[0].imshow(img.crop((468, 212, 768, 512)))
# axes.ravel()[0].title.set_text('Original')
#
# for i, (name, rec) in enumerate(reconstructions.items()):
#     axes.ravel()[i + 1].imshow(rec.crop((468, 212, 768, 512)))  # cropped for easy comparison
#     axes.ravel()[i + 1].title.set_text(name)
#
# plt.show()

# metrics = {}
# filepaths = collect_images("/data/zixinl6/Compress/CLIC_pro/test")
# for f in filepaths:
#     x = read_image(f).unsqueeze(0).to(device)
#     outputs = {}
#     with torch.no_grad():
#         for name, net in networks.items():
#             rv = net(x)
#             rv['x_hat'].clamp_(0, 1)
#             outputs[name] = rv
#     reconstructions = {name: transforms.ToPILImage()(out['x_hat'].squeeze())
#                        for name, out in outputs.items()}
#
#     for name, out in outputs.items():
#         metrics[name] = {
#             'psnr': compute_psnr(x, out["x_hat"]),
#             'ms-ssim': compute_msssim(x, out["x_hat"]),
#             'bit-rate': compute_bpp(out),
#         }
# for k, v in metrics.items():
#     metrics[k] = v / len(filepaths)
header = f'{"Model":30s} | {"t2m_NMSE":<9s} | {"v10_NMSE":<9s} |  {"t2m_v10_NMSE":<9s} | {"PSNR [dB]"} | {"MS-SSIM":<9s} | {"Bpp":<9s}'
print('-'*len(header))
print(header)
print('-'*len(header))
for name, m in metrics.items():
    print(f'{name:35s}', end='')
    for v in m.values():
        print(f' | {v:9.5f}', end='')
    print('|')
print('-'*len(header))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
for name, m in metrics.items():
    axes[0].plot(m['bpp'], m['psnr'], 'o', label=name)
    axes[0].legend(loc='best')
    axes[0].grid()
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Bit-rate [bpp]')
    axes[0].title.set_text('PSNR comparison')

    axes[1].plot(m['bpp'], -10*np.log10(1-m['msssim']), 'o', label=name)
    axes[1].legend(loc='best')
    axes[1].grid()
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Bit-rate [bpp]')
    axes[1].title.set_text('MS-SSIM (log) comparison')

plt.show()