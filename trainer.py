import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time

print("Brain Scan Trainer")
stamp = lambda: int(time.time())

nlen = 52
img = [Image.open(f'inputs/{i}.png') for i in range(nlen)]

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])


inputs = [transform(u) for u in img]
outputs = list(map(lambda tb: int(tb.replace('\n','').split('.')[0]), open('output.txt','r').readlines()))

choices = {0:'normal',1:'tumor',2:'truama',3:'schizophrenic'}

class Predict(nn.Module):
    def __init__(self):
        super(Predict, self).__init__()
        self.moving = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 300, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(300, 150, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.ending = nn.Sequential(
            nn.Linear(150, nlen)
        )
    def forward(self, x):
        x = self.moving(x)
        x = x.flatten(1)
        return self.ending(x)
            

model = Predict()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 300

print("Let the training begin!")
for epoch in range(1, epochs+1):
    rmse = 0
    t0 = stamp()
    for i in range(nlen):
        tout = model(inputs[i].unsqueeze(0))
        tyes = torch.tensor([i], dtype=torch.long)
        loss = criterion(tout, tyes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rmse += loss.item()
    t1 = stamp()
    print(f'Epoch: {epoch} | Loss: {rmse} | Duration: {t1 - t0}')

torch.save(model, 'brain_model.pth')
print("Training has ended, model has been saved")
