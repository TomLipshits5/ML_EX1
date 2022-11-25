# from matplotlib import plot
import random

import nearest_neighbour as nn
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use('seaborn-whitegrid')


def current_milli_time():
    return round(time.time() * 1000)


# global full train and test samples
data = np.load('mnist_all.npz')

train2 = data['train2']
train3 = data['train3']
train5 = data['train5']
train6 = data['train6']

test2 = data['test2']
test3 = data['test3']
test5 = data['test5']
test6 = data['test6']


def getTrainSample(sampleSize: int):
    x_train, y_train = nn.gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], sampleSize)
    return x_train, y_train


def getTestSample(testSize: int):
    x_test, y_test = nn.gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], testSize)
    return x_test, y_test


def predictAndCalcError(k: int, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
    startTime = current_milli_time()
    classifier = nn.learnknn(k, x_train, y_train)
    preds = nn.predictknn(classifier, x_test)
    print('nn.predictknn: {0}'.format(current_milli_time() - startTime))
    return np.mean(y_test.reshape((y_test.shape[0], 1)) != preds)


def corruptSample(x_train, y_train):
    randomSampleIdx = random.sample(range(len(x_train)), int(0.15*len(x_train)))
    for idx in randomSampleIdx:
        y_train[idx] = random.choice([i for i in [2, 3, 5, 6] if i != y_train[idx]])
    return x_train, y_train


def runTestCase(x_test: np.array, y_test: np.array, Xs: list, Ks: list, sampleSizes: list,
                reps=10, corrupt=False, title='title', xlable='Xs', ylable='Ys'):
    errors = [np.zeros(10, float) for _ in Xs]
    for idx, sampleSize in enumerate(Xs):
        print('idx: {}'.format(idx))
        for i in range(reps):
            x_train, y_train = getTrainSample(sampleSizes[idx])
            if corrupt:
                x_train, y_train = corruptSample(x_train, y_train)
            errors[idx][i] = predictAndCalcError(Ks[idx], x_train, y_train, x_test, y_test)
    plt.plot(Xs, [error.mean() for error in errors], label='average error')
    plt.bar(Xs, [error.max() for error in errors], label='max error')
    plt.bar(Xs, [error.min() for error in errors], label='min error')
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.legend()
    plt.show()


def task2a():
    # init sample sizes and errors list
    sampleSizes = [i*10 for i in range(1, 11)]
    Ks = [1 for _ in sampleSizes]
    x_test, y_test = getTestSample(len(test2) + len(test3) + len(test5) + len(test6))
    runTestCase(x_test, y_test, sampleSizes, Ks, sampleSizes,
                title="Task 2a", xlable="train sample size", ylable="error")


def task2e():
    Ks = np.arange(1, 12)
    sampleSizes = [200 for _ in Ks]
    x_test, y_test = getTestSample(len(test2) + len(test3) + len(test5) + len(test6))
    runTestCase(x_test, y_test, Ks, Ks, sampleSizes,
                title="Task 2e", xlable="K value", ylable="error")


def task2f():
    Ks = np.arange(1, 12)
    sampleSizes = [200 for _ in Ks]
    x_test, y_test = getTestSample(len(test2) + len(test3) + len(test5) + len(test6))
    runTestCase(x_test, y_test, Ks, Ks, sampleSizes,
                corrupt=True, title="Task 2f", xlable="K value", ylable="error")


if __name__ == '__main__':
    # task2a()
    # task2e()
    task2f()
