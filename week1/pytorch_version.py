import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Conv2d(3,128,3,2)
        self.layer2 = nn.BatchNorm2d(128)
        self.layer3 = nn.MaxPool2d(2)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Conv2d(128,32,1,1)
        self.layer6 = nn.BatchNorm2d(32)
        self.layer7 = nn.ReLU()
        self.layer8 = nn.Conv2d(32,32,3,2)
        self.layer9 = nn.BatchNorm2d(32)
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Conv2d(32,8,3,1)
        self.layer12 = nn.BatchNorm2d(8)
        self.layer13 = nn.MaxPool2d(2)
        self.layer14 = nn.ReLU()
        self.layer15 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = torch.squeeze(x)

        return x

# Show the example images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    params = {
        # MODEL,
        "batch_size": 16,
        #"optimizer": "SGD",
        "lr": 4.300e-05,
        "momentum": 0.879961,
        "decay": 8.946e-05,

        "rescale": 1.0,
        "zoom": 0.077509,
        "shear": 0.000586,
        "hflip": True,
        #"callback": callbacks,
    }

    print("Running With:")
    print(params)

    show_example = False

    #load the data of train and test
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    train_path = "MIT_split/train"
    train_set = torchvision.datasets.ImageFolder(train_path,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = params["batch_size"], shuffle = True)

    test_path = "MIT_split/test"
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params["batch_size"], shuffle=False)

    # Show the example images
    if show_example:
        # get some random training images
        dataiter = iter(train_loader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % train_loader.dataset.classes[labels[j]] for j in range(params["batch_size"])))

    # Define network
    net = Net()
    net.cuda()

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=params["lr"], momentum=params["momentum"], weight_decay=params["decay"])
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train the network
    for epoch in range(40):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:
                print('[%d, %5d] loss: %.3f' %(epoch+1,i+1,running_loss/5))
                running_loss = 0.0

    print('Finished training')

    #save the trained model
    PATH = './trained_model.pth'
    torch.save(net.state_dict(), PATH)
    # if want to load
    # net = Net()
    # net.load_state_dict(torch.load(PATH))

    if show_example:
        dataiter = iter(test_loader)
        images, labels = dataiter.next()

        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % test_loader.dataset.classes[labels[j]] for j in range(params["batch_size"])))

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % test_loader.dataset.classes[predicted[j]]
                                      for j in range(params["batch_size"])))

    # perform on the whole test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the test dataset is %d %%' %(100*correct/total))

    # calculate the accuracy of each class
    class_correct = list(0. for i in range(8))
    class_total = list(0. for i in range(8))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(8):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(8):
        print('Accuracy of %5s : %2d %%' % (
            test_loader.dataset.classes[i], 100 * class_correct[i] / class_total[i]))
