from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

if __name__ == "__main__":
    
    train_data = pd.read_csv('./dataset_q1/mnist_train.csv', header=None)
    test_data = pd.read_csv('./dataset_q1/mnist_test.csv', header=None)
    
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    x_train, y_train = train_data[:, 1:], train_data[:, 0]
    x_test, y_test = test_data[:, 1:], test_data[:, 0]
    
    activation = "identity"

    clf = MLPClassifier(
        solver='sgd', 
        alpha=0, 
        hidden_layer_sizes=(256, 128, 64), 
        activation=activation, 
        batch_size=200, 
        learning_rate_init=0.01, 
        max_iter=100, 
        verbose=True
    )

    if(activation != "logistic"):
        x_train = normalize(x_train, axis=0, norm='max')
        x_test = normalize(x_test, axis=0, norm='max')
    
    clf.fit(x_train, y_train)
    
    print("Test Accuracy for " + str(activation).upper() + ", sklearn", clf.score(x_test, y_test))