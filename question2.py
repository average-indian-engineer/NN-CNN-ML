import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.svm import SVC 
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.layer2 = nn.MaxPool2d(2)
        self.layer3 = nn.Linear(14 * 14 * 8, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.log_softmax(self.layer6(x), dim=1)
        return x

if __name__ == "__main__":

    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5)
        ),
    ])

    trainset = datasets.FashionMNIST(root = "./dataset_q2", train = True, download = False, transform = transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True)

    testset = datasets.FashionMNIST(root = "./dataset_q2", train = False, download = False, transform = transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = CNN()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 50
    trainLoss = []
    testLoss = []
    for e in range(epochs):
        
        running_loss_train = 0
        running_loss_test = 0
        
        for images, labels in trainloader:    
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
        else:
            print(f"Training loss: {running_loss_train/len(trainloader)}")

        for images, labels in testloader:    
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss_test += loss.item()

        trainLoss.append(running_loss_train/len(trainloader))
        testLoss.append(running_loss_test/len(testloader))

    plt.plot(list(range(1, epochs + 1)), trainLoss, label="Train Loss")
    plt.plot(list(range(1, epochs + 1)), testLoss, label="Test Loss")
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss vs Epochs')
    plt.savefig("traintestloss.png")
    plt.show()

    X_train = np.array([])
    y_train = np.array([])
    setter = True
    for images, labels in trainloader:
        logits = model.forward(images)
        ps = F.softmax(logits, dim=1)
        inpX = ps.detach().numpy()
        inpY = labels.detach().numpy().reshape((batch_size, 1))
        if(setter):
            setter = False
            X_train = inpX
            y_train = inpY
        else:
            X_train = np.vstack((X_train, inpX))
            y_train = np.vstack((y_train, inpY))
    
    cm = confusion_matrix(y_train, np.argmax(X_train, axis=1)) 
    df_cm = pd.DataFrame(cm / cm.sum(axis=1), range(10), range(10))
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
    plt.savefig("confustionmatrixtrain.png")
    plt.show()

    X_test = np.array([])
    y_test = np.array([])
    setter = True
    for images, labels in testloader:
        logits = model.forward(images)
        ps = F.softmax(logits, dim=1)
        inpX = ps.detach().numpy()
        inpY = labels.detach().numpy().reshape((batch_size, 1))
        if(setter):
            setter = False
            X_test = inpX
            y_test = inpY
        else:
            X_test = np.vstack((X_test, inpX))
            y_test = np.vstack((y_test, inpY))

    # np.savetxt("X_train.txt", X_train)
    # np.savetxt("y_train.txt", y_train)
    # np.savetxt("X_test.txt", X_test)
    # np.savetxt("y_test.txt", y_test)

    # X_train = np.loadtxt("X_train.txt")
    # y_train = np.loadtxt("y_train.txt")
    # X_test = np.loadtxt("X_test.txt")
    # y_test = np.loadtxt("y_test.txt")
    
    print(f"Training Accuracy: {np.mean(y_train.ravel() == np.argmax(X_train, axis=1))}")
    print(f"Testing Accuracy: {np.mean(y_test.ravel() == np.argmax(X_test, axis=1))}")

    cm = confusion_matrix(y_test, np.argmax(X_test, axis=1)) 
    df_cm = pd.DataFrame(cm / cm.sum(axis=1), range(10), range(10))
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
    plt.savefig("confustionmatrixtest.png")
    plt.show()

    svm_model_linear = SVC(gamma="auto", kernel = 'rbf', C = 1)
    svm_model_linear.fit(X_train, y_train.ravel())

    trainAccuracy = svm_model_linear.score(X_train, y_train.ravel())    
    print("Train Accuracy SVM: ", trainAccuracy)
    testAccuracy = svm_model_linear.score(X_test, y_test.ravel())
    print("Test Accuracy SVM: ", testAccuracy)

    svm_predictions = svm_model_linear.predict(X_train)
    cm = confusion_matrix(y_train.ravel(), svm_predictions)  
    df_cm = pd.DataFrame(cm / cm.sum(axis=1), range(10), range(10))
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
    plt.savefig("confustionmatrixtrainSVM.png")
    plt.show()

    svm_predictions = svm_model_linear.predict(X_test)
    cm = confusion_matrix(y_test.ravel(), svm_predictions)  
    df_cm = pd.DataFrame(cm / cm.sum(axis=1), range(10), range(10))
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
    plt.savefig("confustionmatrixtestSVM.png")
    plt.show()
