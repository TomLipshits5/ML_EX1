from matplotlib import plot
import nearest_neighbour as nn
import numpy as np

def runTestCase(m: int, reps: int):
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']
    sum = 0

    for i in range(reps):
        x_train, y_train = nn.gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], m)
        x_test, y_test = nn.gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)
        classifer = nn.learnknn(1, x_train, y_train)
        preds = nn.predictknn(classifer, x_test)



