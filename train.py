import torch
import os
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim
from data.triplet_loader import TripletFileLoader
from model import TripletNet, ComplexIrisNet, FeatNet
from model import ExtendedTripletLoss, TripletLoss
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Set the configuration for the training.")
parser.add_argument('--base_path', type=str, default='/home/ra1/Project/Data/UBIRIS/Norm_En/L')
parser.add_argument('--mask_path', type=str, default='/home/ra1/Project/Data/UBIRIS/Mask_Norm/L')
parser.add_argument('--filename','-F', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--dataset', type=str, default='ubiris')
parser.add_argument('--checkpoint', type=str, default= None)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename)


if __name__=="__main__":
    args=parser.parse_args()
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((64,256)),
                        ])
    
    model = ComplexIrisNet()
    
    if args.checkpoint is not None:
        statedict = torch.load(args.checkpoint)['state_dict']
        statedictNew = dict()
        for key, val in statedict.items():
            key = key.replace('module.embeddingnet.', '')
            statedictNew[key] = val
        model.load_state_dict(statedictNew)
        print("Loaded checkpoint '{}'".format(args.checkpoint))
    model = torch.nn.DataParallel(TripletNet(model))    
    
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    # scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    model.train()
    criteria = ExtendedTripletLoss()
    # criteria = TripletLoss()
    criteria.cuda()
    trainset = TripletFileLoader(args.base_path, args.mask_path, args.filename, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
    fileTemp = "./snapshot/"+args.dataset+'/'+args.dataset+'_'+str(0)+"_pth.tar"
    save_checkpoint({'epoch': 0 + 1, 'state_dict': model.state_dict()}, 0, filename=fileTemp)
    for epoch in range(1):
        losses = []
        total_loss = 0
        # trainset = trainset.resample()
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
        for batch_idx, (imgA, imgP, imgN, maskA, maskP, maskN) in enumerate(trainloader):
            optimizer.zero_grad()
            imgA, imgP, imgN, maskA, maskP, maskN = imgA.cuda(), imgP.cuda(), imgN.cuda(), maskA.cuda(), maskP.cuda(), maskN.cuda()
            outputA, outputP, outputN = model(imgA, imgP, imgN)
            loss = criteria(outputA, outputP, outputN, maskA, maskP, maskN) 
            # loss = criteria(outputA, outputP, outputN)           
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            total_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * args.batch_size, len(trainloader.dataset),
                100. * batch_idx / len(trainloader), np.mean(losses))
                print(message)
                losses = []
        print(f'Total loss in Epoch {epoch} is {total_loss/len(trainloader)}')
        # if (epoch+1)%5 == 0:
        fileTemp = "./snapshot/"+args.dataset+'/'+args.dataset+'_'+str(epoch)+"_pth.tar"
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()}, 0, filename=fileTemp)
    
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

    for epoch in range(1, 2):
        losses = []
        total_loss = 0
        # trainset = trainset.resample()
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=20,
        # pin_memory=True) 
        for batch_idx, (imgA, imgP, imgN, maskA, maskP, maskN) in enumerate(trainloader):
            optimizer.zero_grad()
            imgA, imgP, imgN, maskA, maskP, maskN = imgA.cuda(), imgP.cuda(), imgN.cuda(), maskA.cuda(), maskP.cuda(), maskN.cuda()
            outputA, outputP, outputN = model(imgA, imgP, imgN)
            loss = criteria(outputA, outputP, outputN, maskA, maskP, maskN) 
            # loss = criteria(outputA, outputP, outputN)           
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            total_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * args.batch_size, len(trainloader.dataset),
                100. * batch_idx / len(trainloader), np.mean(losses))
                print(message)
                losses = []
        print(f'Total loss in Epoch {epoch} is {total_loss/len(trainloader)}')
        # if (epoch+1)%5 == 0:
        fileTemp = "./snapshot/"+args.dataset+'/'+args.dataset+'_'+str(epoch)+"_pth.tar"
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()}, 0, filename=fileTemp)
        

            
