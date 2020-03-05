# import
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt


class Ffnn:
    X = pd.DataFrame()
    Xhat = pd.DataFrame()
    Y = pd.DataFrame()
    I = 0   # training sets
    N = 0   # features
    J = 3   # output
    K = 7   # hidden values
    loopNum = 1000
    alphaW = 2.3e-4
    alphaV = 1.2e-4
    nbColumns = []
    E = 0   # Error
    EThreshold = 9
    G = pd.DataFrame()
    F = []
    Fhat = pd.DataFrame()

    # def __init__():
    #     pass

    def readData(self):
        # put some title to the data and separate the data in an array
        data = pd.read_csv("data_ffnn_3classes.txt",
                           delimiter=" ", names=["X1", "X2", "Y"])
        # separate the data inputs in a different array, as the outputs
        self.X = data.iloc[:, 0:2]
        # nb of lignes
        self.I = self.X.shape[0]
        # nb of columns
        self.N = self.X.shape[1]
        # the output
        y = (data.iloc[:, 2:3]).values
        self.Y = [[1, 0, 0] if i == 1 else [0, 1, 0]
                  if i == 2 else [0, 0, 1] for i in y]

        self.nbColumns = [str(i) for i in np.ones(self.K)]

    def learn(self):
        self.readData()

        # creation of a matrix of random values
        # print(V)
        V = np.random.uniform(low=-1, high=1, size=(self.N+1, self.K))
        W = np.random.uniform(low=-1, high=1, size=(self.K+1, self.J))

        e_values = []
        min_e = 100

        # while self.E < self.EThreshold:
        for i in range(self.loopNum):
            # print(V)
            Fhat = self.forward_prop(V, W)
            V, W = self.backward_prop(V, W)
            self.sse()
            if self.E < min_e:
                min_e = self.E
            e_values = np.append(e_values, [self.E], axis=0)

        print(min_e)
        plt.plot(e_values)
        plt.show()
        plt.close()

    def forward_prop(self, V, W):
        # addding a columns of 1
        ones = pd.DataFrame(np.ones(self.I), columns=["X0"])
        self.Xhat = pd.concat([ones, self.X], axis=1)

        # multiplication of matrices
        X2hat = np.dot(self.Xhat.values, V)

        # activation fun
        self.F = pow((1 + np.exp(-X2hat)), -1)

        Fdf = pd.DataFrame(self.F, columns=list(self.nbColumns))
        self.Fhat = pd.concat([ones, Fdf], axis=1)

        F2hat = np.dot(self.Fhat.values, W)

        # activation fun
        self.G = pow(1+np.exp(-F2hat), -1)

    def sse(self):
        # calculate of the SSE
        self.E = np.sum((self.G - self.Y) ** 2) / 2
        # print(self.E)

    def backward_prop(self, V, W):
        Fhatnp = self.Fhat.to_numpy()
        # calculate of Wkj
        # sum avant le learning rate
        for k in range(self.K+1):
            W[[k], :] -= np.sum((self.G-self.Y)*self.G*(1-self.G) *
                                Fhatnp[:, [k]]) * self.alphaW
        # print((W[[k], :]).shape)
        # print((W[0][0]).shape)

        Xhatnp = self.Xhat.to_numpy()
        # calcul of Vnk
        for n in range(0, self.N+1):
            for k in range(0, self.K):
                V[n][k] += np.sum((self.G-self.Y)*self.G*W[[k], :] *
                                  self.F[:, [k]]*(1-self.F[:, [k]])*Xhatnp[:, [n]]) * self.alphaV
        return V, W


ffnn = Ffnn()
ffnn.learn()
print(ffnn.X)

# print(V)
