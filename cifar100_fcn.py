import torch
import torchvision
import torchvision.transforms as transforms

torch.cuda.empty_cache()
import gc
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"running on {device}")
print(torch.cuda.device_count())
from torchvision.transforms.transforms import RandomRotation
transform = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Resize(40),
     transforms.RandomHorizontalFlip(),
    #  transforms.RandomRotation(degrees=(-35,35)),
     transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

batch_size = 5 

trainset = torchvision.datasets.CIFAR100(root='files/', train=True,
                                        download=True, transform=transform)

train_set, val_set = torch.utils.data.random_split(trainset, [40000, 10000])

validationloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR100(root='files/', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)


classes = ('apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',' bowl',' boy',' bridge',' bus',' butterfly',' camel',
          ' can',' castle',' caterpillar',' cattle',' chair',' chimpanzee',' clock',' cloud',' cockroach',' couch',' cra',' crocodile',' cup',' dinosaur',' dolphin',
          ' elephant',' flatfish',' forest',' fox',' girl',' hamster',' house',' kangaroo',' keyboard',' lamp',' lawn_mower',' leopard',' lion',' lizard',' lobster',' man',
          ' maple_tree',' motorcycle',' mountain',' mouse',' mushroom',' oak_tree',' orange',' orchid',' otter',' palm_tree',' pear',' pickup_truck',' pine_tree',' plain',' plate',
          ' poppy',' porcupine',' possum',' rabbit',' raccoon',' ray',' road',' rocket',' rose',' sea',' seal',' shark',' shrew',' skunk',' skyscraper',
          'snail',' snake',' spider',' squirrel',' streetcar',' sunflower',' sweet_pepper',' table',' tank',' telephone',' television',
          ' tiger',' tractor',' train',' trout',' tulip',' turtle',' wardrobe',' whale',' willow_tree',' wolf',' woman',' worm')

for batch_idx, data in enumerate(trainloader):
    image, label = data
    print(image.size())
    print(label.size())
    # print('batch: {}\tdata: {}'.format(batch_idx, data))
    break


"""Fatter net"""

from torch.nn.modules.pooling import AdaptiveAvgPool2d
import torch.nn as nn
import torch.nn.functional as F
from OpticalConv2d import OpticalConv2d

def conv_block(in_channels, out_channels, pool=False):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool:
    layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

def optical_conv_block(in_channels, out_channels, pool=False, input_size=32):
  layers = [OpticalConv2d(in_channels, out_channels, kernel_size=3, pseudo_negativity=True, input_size=input_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool:
    layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = optical_conv_block(3,64, input_size=32)
        self.conv2 = optical_conv_block(64,128,pool=True, input_size=32)
        self.res1 = nn.Sequential(optical_conv_block(128,128, input_size=16), optical_conv_block(128,128, input_size=16))
        self.conv3 = optical_conv_block(128,256,pool=True, input_size=16)
        self.conv4 = optical_conv_block(256,512,pool=True, input_size=8)
        self.res2 = nn.Sequential(optical_conv_block(512,512, input_size=4), optical_conv_block(512,512, input_size=4))
        self.conv5 = optical_conv_block(512,1000,pool=True, input_size=4)
        self.res3 = nn.Sequential(optical_conv_block(1000,1000, input_size=2), optical_conv_block(1000,1000, input_size=2))
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(nn.Flatten(), 
                                          nn.Linear(1000, 100))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)+x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)+x
        x = self.conv5(x)
        x = self.res3(x)+x
        x = self.pool(x)
        x = self.classifier(x)
        return x

from tqdm import tqdm
 
def train():
 
  best_val_loss = 10000
  best_val_acc = 0

  for epoch in range(65):  # loop over the dataset multiple times
      
      running_loss = 0.0
      running_accuarcy = 0
      val_loss = 0
      val_acc = 0
      correct = 0
      total = 0
      total_val = 0
      correct_val = 0
      for i, data in enumerate(tqdm(trainloader,desc="Epoch: "+str(epoch+1)), 0):
          
          
          # get the inputs; data is a list of [inputs, labels]
          net.train()
          inputs, labels = data
          inputs,labels = inputs.to(device), labels.to(device)
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels).to(device)
          loss.backward()
          optimizer.step()
          
          # print statistics
          with torch.no_grad():
            running_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_accuarcy = 100 * correct // total
          del inputs, labels

      for val_data in validationloader:
        with torch.no_grad():
          val_images, val_labels = val_data
          val_images,val_labels = val_images.to(device), val_labels.to(device)
          #calculate validation
          # val_images, val_labels = val_images.to(device), val_labels.to(device)
          net.eval()
          y_pred = net(val_images)
          val_loss = criterion(y_pred, val_labels).to(device)
          _, val_predicted_label = torch.max(y_pred.data, 1)
          correct_val += (val_predicted_label == val_labels).sum().item()
          total_val += val_labels.size(0)
      val_acc = 100 * correct_val // total_val
      sch.step()
      print(f'Epoch{epoch + 1}:      loss: {running_loss:.3f} accuaracy: {running_accuarcy}% val_loss: {val_loss:.3f} val_acc: {val_acc}%\n')
      if val_loss<best_val_loss:
        best_val_loss = val_loss
      if val_acc>best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), "bestcifar_net.pth")
      elif val_acc==best_val_acc and val_loss<best_val_loss:
        torch.save(net.state_dict(), "bestcifar_net.pth")
      torch.save(net.state_dict(), "cifar_net.pth")
      del val_images, val_labels 
        
  print('Finished Training')
  return running_accuarcy, best_val_acc

"""Test"""

def evaluate():
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in tqdm_notebook(testloader, desc="evaluation: "):
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          # calculate outputs by running images through the network
          outputs = net(images)
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = 100 * correct // total
  print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
  return accuracy

import torch.optim as optim
import pandas as pd


net = Net()
net.load_state_dict(torch.load('cifar_net.pth'))
# net = nn.DataParallel(net, device_ids=[0,1])
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
sch = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=25,gamma=0.5)
criterion.to(device)

train_acc, val_acc = train()

test_acc = evaluate()
