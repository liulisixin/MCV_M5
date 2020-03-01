import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

import torch.optim as optim

from tqdm import tqdm


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


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()  # set model to training mode

    running_loss = 0
    running_corrects = 0
    total = 0

    for data, labels in loader:
        optimizer.zero_grad()  # make the gradients 0

        x = data.cuda()
        y = labels.cuda()

        output = model(x)  # forward pass
        loss = criterion(output, y)  # calculate the loss value
        preds = output.max(1)[1]  # get the predictions for each sample

        loss.backward()  # compute the gradients
        optimizer.step()  # uptade network parameters

        # statistics
        running_loss += loss.item() * x.size(0)
        # .item() converts type from torch to python float or int
        running_corrects += torch.sum(preds == y).item()
        total += float(y.size(0))

    epoch_loss = running_loss / total  # mean epoch loss
    epoch_acc = running_corrects / total  # mean epoch accuracy

    return epoch_loss, epoch_acc


def test_one_epoch(model, loader, criterion):
    model.eval()  # set model to validation mode

    running_loss = 0
    running_corrects = 0
    total = 0

    # We are not backpropagating through the validation set, so we can save time  and memory
    # by not computing the gradients
    with torch.no_grad():
        for data, labels in loader:
            x = data.cuda()
            y = labels.cuda()

            output = model(x)  # forward pass

            # Calculate the loss value (we do not to apply softmax to our output because Pytorch's
            # implementation of the cross entropy loss does it for us)
            loss = criterion(output, y)
            preds = output.max(1)[1]  # get the predictions for each sample

            # Statistics
            running_loss += loss.item() * x.size(0)
            # .item() converts type from torch to python float or int
            running_corrects += torch.sum(preds == y).item()
            total += float(y.size(0))

    epoch_loss = running_loss / total  # mean epoch loss
    epoch_acc = running_corrects / total  # mean epoch accuracy

    return epoch_loss, epoch_acc,

def plot_history(train_accuracy, test_accuracy, train_loss, test_loss, title = "pic_"):
    # summarize history for accuracy
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(title+'accuracy.png')
    plt.close()
    # summarize history for loss
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(title+'loss.png')
    plt.close()

if __name__ == "__main__":
    params = {
        # MODEL,
        "batch_size": 16,
        #"optimizer": "SGD",
        "lr": 0.001,
        "momentum": 0.9,
    }

    print("Running With:")
    print(params)

    show_example = False

    # load the data of train and test
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform = transforms.Compose([transforms.ToTensor()])

    train_path = "../MIT_split/train"
    train_set = torchvision.datasets.ImageFolder(train_path,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = params["batch_size"], shuffle = True)

    test_path = "../MIT_split/test"
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
    optimizer = optim.SGD(net.parameters(), lr=params["lr"], momentum=params["momentum"])

    # train the network
    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []
    best_test_accuracy = 0.0

    for epoch in tqdm(range(40)):
        loss, acc = train_one_epoch(net, train_loader, criterion, optimizer)
        train_loss.append(loss)
        train_accuracy.append(acc)
        print('[%d] train loss: %.3f train acc: %.3f' % (epoch + 1, loss, acc))


        loss, acc = test_one_epoch(net, test_loader, criterion)
        test_loss.append(loss)
        test_accuracy.append(acc)
        print('[%d] test loss: %.3f test acc: %.3f' % (epoch + 1, loss, acc))
        if acc > best_test_accuracy:
            best_test_accuracy = acc

    plot_history(train_accuracy, test_accuracy, train_loss, test_loss)
    print('Best acc: %.3f' % (best_test_accuracy))


    """
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

    
    # save the trained model
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
        print('GroundTruth: ',
              ' '.join('%5s' % test_loader.dataset.classes[labels[j]] for j in range(params["batch_size"])))

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % test_loader.dataset.classes[predicted[j]]
                                      for j in range(params["batch_size"])))
    
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
    
    """
