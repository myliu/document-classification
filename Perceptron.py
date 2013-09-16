## author: Mingyu Liu
## author: Shi He

import numpy as np
import string
from Dataset import *

def main():
    d = Dataset("rec.sport.hockey.txt", "rec.sport.baseball.txt", cutoff=2000)
    (Xtrain, Ytrain, Xtest, Ytest) = d.getTrainAndTestSets(0.8, seed=100)
    w = np.asmatrix([0 for elem in range(Xtrain.shape[1])])

    learningRate = 1

##  numTrial is the total number of rounds we want to go through before stopping (in case it is not converged)
##  k is to keep track of how many rounds we have been through   
    numTrial = 5
    k = 0

##  wSum is to count the sum of w in a given round
##  wAvg is to count the avg of w in a given round
    wAvg = w
    while makeError(Xtrain, Ytrain, wAvg):
        
        if k >= numTrial:
            print "No perfect hyperplane found!"
            print "Stop after " + str(numTrial) + " iterations."
            break
        k += 1
        
        for i in range(Xtrain.shape[0]):
            expected = -1
            xtrain = np.asmatrix(Xtrain[i]).T
            if w * xtrain > 0:
                expected = 1
            if expected != Ytrain[i]:
                w = w + learningRate * Ytrain[i] * Xtrain[i]
            if i == 0:
                wSum = w
            else:
                wSum += w
        wAvg = wSum / Xtrain.shape[0]

    trainError = computeError(Xtrain, Ytrain, w)
    print 'Train error rate is ' + str(trainError)
    testError = computeError(Xtest, Ytest, w)
    print 'Test error rate is ' + str(testError)

def makeError(Xtrain, Ytrain, w):
    errorRate = computeError(Xtrain, Ytrain, w)
    if (errorRate != 0):
        return True
    else:
        return False

def computeError(Xtest, Ytest, w):
    w = w.T
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



