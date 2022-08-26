import torch
import os
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim
from data.triplet_loader import TripletImageLoader, TestImageLoader
from model import TripletNet
from model import ComplexIrisNet, FeatNet
from model import ExtendedTripletLoss, TripletLoss
import numpy as np
import argparse
from util import plot_roc
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
argparser = argparse.ArgumentParser(description="Set up the configuration for the testing.")
argparser.add_argument('--base_path', type=str,
                    default='../Data/UBIRIS/Norm_En/R')
argparser.add_argument('--mask_path', type=str,
                    default='../Data/UBIRIS/Mask_Norm/R')
argparser.add_argument('--checkpoint', type=str, default= 'snapshot/ubiris/ubiris.pth.tar')
argparser.add_argument('--save_path', type=str, default='./ROC.png')
argparser.add_argument('--resume', action='store_true', default=False)
argparser.add_argument('-d''--division', type=int, default=1)
args = argparser.parse_args()

@torch.no_grad()
def _compute_shift_loss(feat1, mask1, feat2, mask2, shift=4):
    feat1 = feat1.cuda()
    feat2 = feat2.cuda()
    mask1 = mask1.repeat(1, feat1.shape[1], 1, 1).clone().cuda()
    mask2 = mask2.repeat(1, feat2.shape[1], 1, 1).clone().cuda()
    dist = torch.zeros(2*shift+1, feat1.shape[0]).to(feat1.device)
    mr = torch.zeros(2*shift+1, feat1.shape[0]).to(feat1.device)
    for i in range(2*shift+1):
        featm2 = torch.roll(feat2, -shift+i, -1).clone()
        maskm2 = torch.roll(mask2, -shift+i, -1).clone()
        featm1 = feat1.clone()
        maskm1 = mask1.clone()
        mask = torch.bitwise_and(maskm1, maskm2)
        featm1[mask == False] = 0
        featm2[mask == False] = 0
        # Normalize by number of pixels
        dist[i] = torch.norm(
            featm1-featm2, 2, dim=(1, 2, 3))**2/(mask.sum(dim=(1, 2, 3))+1e-3)
        mr[i] = mask.sum(dim=(1, 2, 3))/(256.0*16)
    # print(dist)
    loss, ind = dist.min(0, keepdim=True)
    mrate = torch.gather(mr, 0, ind)
    return loss.detach().cpu().numpy(), mrate.detach().cpu().numpy()

def compute_shift_loss(feat1, mask1, feat2, mask2, division, shift=4):
    if division == 1:
        return _compute_shift_loss(feat1, mask1, feat2, mask2, shift)
    else:
        slicing = int(feat1.shape[0]/division)
        slicing_points = [i*slicing for i in range(division)]
        slicing_points.append(feat1.shape[0])

        for i in range(division):
            loss, mrate = _compute_shift_loss(feat1[slicing_points[i]:slicing_points[i+1]], mask1=mask1[slicing_points[i]:slicing_points[i+1]], feat2=feat2[slicing_points[i]:slicing_points[i+1]], mask2=mask2[slicing_points[i]:slicing_points[i+1]], shift=shift)
            if i == 0:
                loss_all = loss
                mrate_all = mrate
            else:
                loss_all = np.concatenate((loss_all, loss), axis=1)
                mrate_all = np.concatenate((mrate_all, mrate), axis=1)
        return loss_all, mrate_all

@torch.no_grad()
def inference():
    model = ComplexIrisNet()
    model = torch.nn.DataParallel(model)
    model.cuda()
    statedict = torch.load(args.checkpoint)['state_dict']
    statedictNew = dict()
    for key, val in statedict.items():
        key = key.replace('module.embeddingnet.', 'module.')
        statedictNew[key] = val
    model.load_state_dict(statedictNew)
    testLoader = torch.utils.data.DataLoader(TestImageLoader(args.base_path, args.mask_path),
                                            batch_size=16, shuffle=False, num_workers=10, pin_memory=True)

    model.eval()
    featList = list()
    maskList = list()
    labelList = list()
    for batch_idx, (imgs, masks, labels) in tqdm(enumerate(testLoader)):
        imgs, masks = imgs.cuda(), masks.cuda()
        with torch.no_grad():
            feat = model(imgs)
            featList.append(torch.cat((feat.real, feat.imag), dim=1).cpu())
            maskList.append(masks.cpu())
            labelList.append(labels)
    feat = torch.cat(featList, 0).numpy()
    print(f"feat shape is {feat.shape}")
    mask = torch.cat(maskList, 0).numpy()
    print(f"mask shape is {mask.shape}")
    label = np.concatenate(labelList)
    print(f"label shape is {label.shape}")
    np.save('match/feat.npy', feat)
    np.save('match/mask.npy', mask)
    np.save('match/label.npy', label)
    return feat, mask, label

if __name__ == '__main__':
    if not os.path.isdir('match'):
        os.makedirs('match')
    if args.resume:
        feat = np.load('match/feat.npy')
        mask = np.load('match/mask.npy')
        label = np.load('match/label.npy')
    else:
        feat, mask, label = inference()
    feat = torch.from_numpy(feat)
    mask = torch.from_numpy(mask)
    scores = np.zeros((len(feat), len(feat)//2), dtype=np.float16)
    mrates = np.zeros((len(feat), len(feat)//2), dtype=np.float16)
    labelMat = np.zeros(scores.shape, dtype=np.int8)
    for i in tqdm(range(1, len(feat)//2+1)):
        scores[:, i-1], mrates[:, i-1] = compute_shift_loss(
            feat, mask, torch.roll(feat, i), torch.roll(mask, i), args.division)
        labelMat[:, i-1] = (label == np.roll(label, i))
    np.save('match/scores.npy', scores)
    np.save('match/mrates.npy', mrates)
    np.save('match/labelMat.npy', labelMat)
    gg = scores[np.bitwise_and(labelMat == 1, mrates > 0.5)]
    ii = scores[np.bitwise_and(labelMat == 0, mrates > 0.5)]
    plot_roc(gg, ii, saveDir=args.save_path)
    sio.savemat('scores.mat', {'scores':scores, 'mrates':mrates, 'labelMat':labelMat})