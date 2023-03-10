from cProfile import label
import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

#batch maker
def next_batch(inputs,targets,batchSize):
    for i in range(0,inputs.shape[0],batchSize):
        yield(inputs[i:i+batchSize],targets[i:i+batchSize])

#init training parameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

#generate dataset for now
print("[INFO] preparing data...")
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3,cluster_std=2.5, random_state=95)
#create train,test sets
(trainX, testX, trainY, testY) = train_test_split(X, y,test_size=0.15, random_state=95)
#convert to tensors from numpy
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

#initialize the mlp, optimizer and loss
MLP = mlp.get_training_model().to(DEVICE)
print(MLP)
opt = SGD(MLP.parameters(),lr=LR)
lossFunc = nn.CrossEntropyLoss()

#Training loop
trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
train_loss_graph = []
train_acc_graph = []
test_loss_graph = []
test_acc_graph = []
#outer loop for epochs
for epoch in range(0, EPOCHS):
# initialize tracker variables and set our model to trainable
    print("[INFO] epoch: {}...".format(epoch + 1))
    trainLoss = 0
    trainAcc = 0
    samples = 0
    batch=0
    MLP.train()
    for (batchX,batchY) in next_batch(trainX,trainY, BATCH_SIZE):
        #flash to device and calculate loss
        (batchX,batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
        predictions = MLP(batchX)
        loss = lossFunc(predictions,batchY.long())
        #zero the grad, do backprop, update weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        #update train loss,accuracy,visited samples
        trainLoss += loss.item() * batchY.size(0)
        trainAcc += (predictions.max(1)[1] == batchY).sum().item()
        samples += batchY.size(0)
        batch+=1
        print("{} : current train batch".format(batch))
        print(trainTemplate.format(epoch + 1, (trainLoss / samples),(trainAcc / samples)))
    train_loss_graph.append(trainLoss / samples)
    train_acc_graph.append(trainAcc / samples)
    #test mode for the epoch
    testLoss = 0
    testAcc = 0
    samples = 0
    batch = 0
    MLP.eval()
    with torch.no_grad():
        for (batchX,batchY) in next_batch(testX,testY,BATCH_SIZE):
            (batchX,batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
            predictions = MLP(batchX)
            loss = lossFunc(predictions,batchY.long())
            testLoss += loss.item() * batchY.size(0)
            testAcc += (predictions.max(1)[1] == batchY).sum().item()
            samples += batchY.size(0)
            batch+=1
            print("{} : current test batch".format(batch))
            testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
            print(testTemplate.format(epoch + 1, (testLoss / samples),(testAcc / samples)))
        print("")
    test_loss_graph.append(testLoss / samples)
    test_acc_graph.append(testAcc / samples)

fig, ax = plt.subplots(1,2)

ax[0].plot(train_loss_graph,'r',label="train-loss")
ax[0].plot(test_loss_graph,'b',label="test-loss")
ax[0].legend()
ax[1].plot(train_acc_graph,'g',label="train-acc")
ax[1].plot(test_acc_graph,'k',label="test-acc")
ax[1].legend()
plt.show()



