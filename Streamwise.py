## author: Mingyu Liu
## author: Shi He

import numpy as np
import string
from Dataset import *


def main():
    d = Dataset("rec.sport.hockey.txt", "rec.sport.baseball.txt", cutoff=1000)
    (Xtrain, Ytrain, Xtest, Ytest) = d.getTrainAndTestSets(0.8, seed=100)

    lam = 100
    cols = []
    currentError = 1
    n = Xtrain.shape[1]
    dic = {}

    for j in range(n):
        cols.append(j)     
        w = trainRidge(Xtrain[:, cols], Ytrain, lam)
        errorRate = computeError(Xtrain[:, cols], Ytrain, w)
        if errorRate >= currentError:
            cols.pop()
        else:
            dic[j] = currentError - errorRate
            currentError = errorRate
##          print out currentError once a while  
            if j % 10 == 0:
                print currentError

    w = trainRidge(Xtrain[:, cols], Ytrain, lam)
    trainError = computeError(Xtrain[:, cols], Ytrain, w)
    print 'Train error rate is ' + str(trainError)
    testError = computeError(Xtest[:, cols], Ytest, w)
    print 'Test error rate is ' + str(testError)

##  find the top 10 features
    wordList = d.getWordList()
    topCols = [(key, value) for key, value in sorted(dic.iteritems(), key = lambda(k, v) : (v, k), reverse = True)]
    topCols = topCols[: 10]
    topFeatures = [wordList[index] for (index, value) in topCols]
    for f in topFeatures:
        print f

def trainRidge(Xtrain, Ytrain, lam):
    Xmatrix = np.asmatrix(Xtrain)
    Ymatrix = np.asmatrix(Ytrain)
    return np.linalg.inv(Xmatrix.T * Xmatrix + lam * np.eye(Xmatrix.shape[1])) * Xmatrix.T * Ymatrix

def computeError(Xtest, Ytest, w):
    correctCount = 0
    incorrectCount = 0
    for testIndex in range(Ytest.size):
        xtest = Xtest[testIndex]
        xtestw = xtest * w

##      if sign(xtestw) > 0, expected = 1; if sign(xtestw) <= 0, expected = -1        
        expected = -1
        if xtestw > 0:
            expected = 1
            
        if expected == Ytest[testIndex]:
            correctCount += 1
        else:
            incorrectCount += 1
    return incorrectCount * 1.0 / (correctCount + incorrectCount)
                
if __name__ == "__main__":
    main()



