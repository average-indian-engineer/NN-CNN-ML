import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from progressbar import ProgressBar, Percentage, Bar, ETA
import pickle

class Neural_Net:
    
    def __init__(self, N, nodes, activation_function, learning_rate, visualize):
        self.N = N
        self.nodes = nodes
        self.activation_function = self.getActivationFunction(activation_function)
        self.deactivation_function = self.getDeactivationFunction(activation_function)
        self.learning_rate  = learning_rate
        self.weights = []
        self.bias = []
        self.initialize()
        self.normalize = True
        self.visualize = visualize
        self.activation_name = activation_function.lower()
        if(activation_function.lower() == "sigmoid"):
            self.normalize = False
        
    
    def initialize(self):
        for i in range(self.N - 1):
            weights = np.random.normal(0, 1, size=(self.nodes[i], self.nodes[i+1]))
            self.weights.append(weights * 0.01)
            bias = np.zeros((1, self.nodes[i+1]))
            self.bias.append(bias)
    
    def getActivationFunction(self, activation_function):
        stringToMatch = activation_function.lower()
        if(stringToMatch == "relu"):
            return self.relu
        elif(stringToMatch == "sigmoid"):
            return self.sigmoid
        elif(stringToMatch == "linear" or stringToMatch == "identity"):
            return self.linear
        elif(stringToMatch == "tanh"):
            return self.tanh
    
    def relu(self, x):
        return np.where(x < 0, 0, x)
        
    def sigmoid(self, x):
        return .5 * (1 + np.tanh(.5 * x))

    def linear(self, x):
        return x

    def tanh(self, x):
        return np.tanh(x)

    def getDeactivationFunction(self, activation_function):
        stringToMatch = activation_function.lower()
        if(stringToMatch == "relu"):
            return self.reluD
        elif(stringToMatch == "sigmoid"):
            return self.sigmoidD
        elif(stringToMatch == "linear" or stringToMatch == "identity"):
            return self.linearD
        elif(stringToMatch == "tanh"):
            return self.tanhD
    
    def reluD(self, x):
        return np.where(x == 0, 0, 1)
    
    def sigmoidD(self, x):
        return x * (1 - x)

    def linearD(self, x):
        return 1

    def tanhD(self, x):
        return 1.0 - np.power(x, 2)
    
    def softmax(self, x):
        x -= np.max(x)
        expX = np.exp(x)
        return expX / np.sum(expX, axis = 1, keepdims = True)
    
    def fit(self, X, y, epochs, batch_size):
        
        batchCounts = X.shape[0] // batch_size
        if(self.normalize):
            X = normalize(X, axis=0, norm='max')
        X, y = shuffle(X, y)

        trainError = []
        trainCount = []

        for epoch in range(epochs):
            pbar = ProgressBar(widgets=["Epoch: " + str(epoch + 1) + " ", Percentage(), Bar(), ETA()], maxval=batchCounts).start()
            for batch in range(batchCounts):
                
                x_train_batch = X[batch * batch_size : (batch+1) * batch_size]
                y_train_batch = y[batch * batch_size : (batch+1) * batch_size]

                ys = []
                ys.append(x_train_batch)
                
                # ForwardFeed
                for j in range(self.N - 1):
                    vj = (ys[j] @ self.weights[j]) + self.bias[j]
                    if(j < self.N - 2):
                        activatedOutputs = self.activation_function(vj)
                    else:
                        activatedOutputs = self.softmax(vj)
                    ys.append(activatedOutputs)

                real_output = np.zeros((batch_size, self.nodes[-1]))
                output_layer = ys[self.N - 1]
                
                for j in range(batch_size):
                    real_output[j, y_train_batch[j]] = 1
                        
                dWeights = [None for i in range(self.N - 1)]
                dBias = [None for i in range(self.N - 1)]
                
                # BackFeed
                db = output_layer - real_output
                dw = ys[self.N - 2].T @ db
                dWeights[self.N - 2] = dw
                dBias[self.N - 2] = db
                for j in range(self.N - 2, 0, -1):
                    db = (dBias[j] @ self.weights[j].T) * self.deactivation_function(ys[j])
                    dBias[j-1] = db
                    dWeights[j-1] = ys[j-1].T @ (db)

                updateWeightsWith = []    
                updateBiasWith = []  

                for j in range(self.N - 1):
                    updateWeightsWith.append((self.learning_rate / batch_size) * dWeights[j])
                    updateBiasWith.append((self.learning_rate / batch_size) * dBias[j].sum(axis = 0))

                for j in range(self.N - 1):
                    self.weights[j] -= updateWeightsWith[j]
                    self.bias[j] -= updateBiasWith[j]
                
                pbar.update(batch + 1)
                    
            trainError.append(1 - self.score(X, y))
            trainCount.append(epoch)
            pbar.finish()

        if(self.visualize):
            plt.plot(trainCount, trainError, 'c')
            plt.ylabel('Training Error')
            plt.xlabel('Epochs')
            plt.title("Training Error vs Epochs with " + str(self.activation_name).upper() + " Activation")
            plt.show()

    def predict(self, X):
        if(self.normalize):
            X = normalize(X, axis=0, norm='max')
        ys = []
        ys.append(X)
        for j in range(self.N - 1):
            vj = (ys[j] @ self.weights[j]) + self.bias[j]
            if(j < self.N - 2):
                activatedOutputs = self.activation_function(vj)
            else:
                activatedOutputs = self.softmax(vj)
            ys.append(activatedOutputs)
        output_layer = ys[self.N - 1]
        return np.argmax(output_layer, axis=1).T
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
if __name__ == "__main__":
    
    train_data = pd.read_csv('./dataset_q1/mnist_train.csv', header=None)
    test_data = pd.read_csv('./dataset_q1/mnist_test.csv', header=None)
    
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # Configure Neural Net
    N = 5
    nodes = [784, 256, 128, 64, 10]
    activation_function = "linear"
    learning_rate = 0.1
    visualize = True

    CNN = Neural_Net(N, nodes, activation_function, learning_rate, visualize)
    CNN.fit(train_data[:,1:].copy(), train_data[:,0].copy(), 100, 200)
    print("Test Accuracy with " + str(activation_function).upper() + ": ", CNN.score(test_data[:,1:].copy(), test_data[:,0].copy()))
    pickle.dump(CNN, open("./" + str(activation_function) + ".dat", 'wb'))

    # CNN = pickle.load(open('./Saved Models/linear.dat', 'rb'))
    # print(CNN.weights)