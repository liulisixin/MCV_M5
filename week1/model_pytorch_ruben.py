import torch.nn as nn
import torchvision
import torch
import sys
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SeparableConv2d,self).__init__()
        self.deepwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self,x):
        x = self.deepwise(x)
        x = self.pointwise(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.batch = nn.BatchNorm2d(out_channels)
        self.actv = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.actv(x)
        return x

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SeparableConvBlock, self).__init__()
        self.conv = SeparableConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride
        )
        self.batch = nn.BatchNorm2d(out_channels)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.actv(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = ConvBlock(3, 48, (3,3), (1, 1))
        self.block2 = ConvBlock(48, 48, (3,3), (1, 1))
        
        self.maxPooling1 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2, 2)
        )

        self.block3 = ConvBlock(48, 48, (3,3), (1, 1))
        self.block4 = ConvBlock(48, 48, (3,3), (1, 1))

        self.avgPooling1 = nn.AvgPool2d(
            kernel_size=(3,3),
            stride=(2,2)
        )
        
        self.block5 = ConvBlock(48, 62, (3,3), (1, 1))
        self.block6 = ConvBlock(62, 62, (3,3), (1, 1))

        self.avgPooling2 = nn.AvgPool2d(
            kernel_size=(3,3),
            stride=(2,2)
        )
        
        self.block7 = ConvBlock(62, 62, (3,3), (1, 1))
        self.block8 = ConvBlock(62, 62, (3,3), (1, 1))
        
        self.avgPooling3 = nn.AvgPool2d(
            kernel_size=(3,3),
            stride=(2,2)
        )
        
        self.block9 = ConvBlock(62, 62, (3,3), (1, 1))
        self.block10 = ConvBlock(62, 62, (3,3), (1, 1))
        
        self.maxPooling4 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2, 2)
        )
        self.batchNorm = nn.BatchNorm2d(62)
        self.flatt = nn.Flatten()
        self.linear = nn.Linear(558,8)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.maxPooling1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgPooling1(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.avgPooling2(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgPooling3(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.maxPooling4(x)
        x = self.batchNorm(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def create_datasets(img_size, batch_size, train_dataset_path, test_dataset_path):
    # Prepare the datasets
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform)
    training_data = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform)
    test_data = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size
    )
    return training_data, test_data

def generate_optimizer(type, net):
    if type.lower() == 'adam':
        return optim.Adam(net.parameters(), lr=0.001)
    elif type.lower() == 'adadelta':
        return optim.Adadelta(net.parameters())
    raise NotImplementedError

def train_model(train_dataset, net, device, optimizer, criterion):
    train_loss = 0.0
    train_acc = 0.0
    total = 0.0
    for data in train_dataset:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()
        train_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
    return train_loss / total, train_acc / total

def test_model(test_dataset, net, device, criterion):
    val_loss = 0.0
    val_acc = 0.0
    total = 0.0
    with torch.no_grad():
        for data in test_dataset:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            val_acc += (predicted == labels).sum().item()
            val_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
    return val_loss / total, val_acc / total

def plot(history):
    plt.plot(history['acc']['train'])
    plt.plot(history['acc']['val'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy_soft.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history['loss']['train'])
    plt.plot(history['loss']['val'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss_soft.jpg')
    plt.close()

def main():
    # An example how to build and train a CNN: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html    
    
    device = 'cuda' 
    if not torch.cuda.is_available():
        print('No GPU available! Exit ...')
        sys.exit(1)
    
    # Const variables
    train_dataset_path = '/home/mcv/datasets/MIT_split/train/'
    test_dataset_path = '/home/mcv/datasets/MIT_split/test/'
    img_width = 256
    img_height = 256
    batch_size = 32
    number_of_epoch = 50

    train_dataset, test_dataset = create_datasets((img_width, img_height), batch_size, train_dataset_path, test_dataset_path)

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = generate_optimizer('adam', net)

    history = {'acc' : {'train' : [], 'val' : []}, 'loss' : {'train' : [], 'val' : []}}
    for epoch in range(number_of_epoch):  # loop over the dataset multiple times
        train_loss, train_acc = train_model(train_dataset, net, device, optimizer, criterion)
        val_loss, val_acc = test_model(test_dataset, net, device, criterion)
        history['loss']['val'].append(val_loss)
        history['loss']['train'].append(train_loss)
        history['acc']['val'].append(val_acc)
        history['acc']['train'].append(train_acc)
        print('Epoch[{}/{}]\tTrain Loss: {}\tTrain Acc: {}\tVal Loss: {}\tVal Acc: {}'.format(epoch+1, number_of_epoch, train_loss, train_acc, val_loss, val_acc))
    plot(history)

if __name__ == "__main__":
    main()