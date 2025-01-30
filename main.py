import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd


#seperate columns of gt_test
#add labels for image
#Import data





def import_data():
    
    labels = []
    data = []
    content = []

    with open('data/gt_test.txt', 'r') as file:
        for line in file:
            test = line.strip().split(" ", 2)
            data.append(test)   

    print(data)
    return data

def main():
    data = import_data()
   

if __name__ == '__main__':
    main()

# #select device
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

# #define NN
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten() 
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),  
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         x=self.flatten(x)
#         logits=self.linear_relu_stack(x)
#         return logits

# model = NeuralNetwork().to(device)

# try:
#   state_dict = torch.load('model.pt')
#   model.load_state_dict(state_dict)
# except:
#   print("No file found")

# print(model)

# #create data loaders
# batch_size = 64
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# #Training function
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):

#         X, y = X.to(device), y.to(device)

#         #calculate error by comparing expected vs actual
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         #backproopagate to calculate gradient
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
    
#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# #test function
# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X,y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# #training loop
# epochs = 500
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-----------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("done!")

# torch.save(model.state_dict(), 'model.pt')
