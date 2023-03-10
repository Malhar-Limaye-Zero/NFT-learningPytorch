import torch
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from CIFARmodel import Net
from torch.utils.data import random_split
import numpy as np
from sklearn.metrics import classification_report
#hyperparams
batch_size = 64
LR = 0.0005
EPOCHS = 100
TRAIN_SPLIT = 0.85
VAL_SPLIT = 1-TRAIN_SPLIT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
#dataloading
do_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
print("Loading dataset.....")
trainset = torchvision.datasets.CIFAR10(root="data",train=True,download=True,transform=do_transform)
testset = torchvision.datasets.CIFAR10(root="data",train=False,download=True,transform=do_transform)

print("Generating train/valid split.....")
numtrain = int(len(trainset)*TRAIN_SPLIT)
numval = int(len(trainset)*VAL_SPLIT)
(trainset,valset) = random_split(trainset,[numtrain,numval],generator=torch.Generator().manual_seed(42))

traindataloader = DataLoader(trainset,shuffle=True,batch_size=batch_size)
valdataloader = DataLoader(valset,batch_size=batch_size)
testdataloader = DataLoader(testset,batch_size=batch_size)
print("train set size: {}".format(len(traindataloader.dataset)))
print("validator set size: {}".format(len(valdataloader.dataset)))
trainsteps = len(traindataloader.dataset)//batch_size
valsteps = len(valdataloader)//batch_size

#model
net = Net(3,len(trainset.dataset.classes)).to(DEVICE)
opt = optim.Adam(net.parameters(),lr=LR,weight_decay=1e-3)
lossFn = torch.nn.NLLLoss()

H = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
print("Starting training")
for e in range(0,EPOCHS):
    net.train()
    trainLoss = 0
    valLoss = 0
    trainCorrect = 0
    valCorrect = 0
    for(x,y) in traindataloader:
        (x,y) = (x.cuda(),y.cuda())
        pred=net(x)
        loss = lossFn(pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        trainLoss+=loss
        trainCorrect+=(pred.argmax(1)==y).type(torch.float).sum().item()
    
    with torch.no_grad():
        net.eval()
        for(x,y) in valdataloader:
            (x,y) = (x.cuda(),y.cuda())
            pred = net(x)
            valLoss+=lossFn(pred,y)
            valCorrect+=(pred.argmax(1)==y).type(torch.float).sum().item()
    
    avgtrainloss = trainLoss/trainsteps
    avgvalloss = valLoss/valsteps
    trainCorrect = trainCorrect/len(traindataloader.dataset)
    valCorrect = valCorrect/len(valdataloader.dataset)
    H["train_loss"].append(avgtrainloss.cpu().detach().numpy())
    H["val_loss"].append(avgvalloss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_acc"].append(valCorrect)
    
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgtrainloss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgvalloss, valCorrect))

print("[INFO] evaluating network...")
with torch.no_grad():
    net.eval()
    preds=[]
    for(x,y) in testdataloader:
        x=x.cuda()
        pred=net(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

print(classification_report(np.array(testset.targets),np.array(preds),target_names=testset.classes))