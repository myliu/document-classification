## author: Mingyu Liu
## author: Shi He

import numpy as np
import string
from Dataset import *


def main():
    d = Dataset("rec.sport.hockey.txt", "rec.sport.baseball.txt", cutoff=2000)
    (Xtrain, Ytrain, Xtest, Ytest) = d.getTrainAndTestSets(0.8, seed=100)

    pC1 = getClassProb(Ytrain, -1)
    pC2 = getClassProb(Ytrain, 1)

    wordList = d.getWordList()
    w1 = [getFeatureProb(Xtrain, Ytrain, -1, wordIndex) for wordIndex in range(len(wordList))]
    aw1 = np.asarray(w1)
    w2 = [getFeatureProb(Xtrain, Ytrain, 1, wordIndex) for wordIndex in range(len(wordList))]
    aw2 = np.asarray(w2)

    trainError = computeError(Xtrain, Ytrain, pC1, pC2, aw1, aw2)
    print 'Train error rate is ' + str(trainError)
    testError = computeError(Xtest, Ytest, pC1, pC2, aw1, aw2)
    print 'Test error rate is ' + str(testError)

def computeError(Xtest, Ytest, pC1, pC2, aw1, aw2):
    correctCount = 0
    incorrectCount = 0
    for testIndex in range(Ytest.size):
        xtest = Xtest[testIndex]
##      C = argmax P(x1, x2, ..., xn | C) P(C)
##      expected is C
##      np.sum(aw1 * xtest) is P(x1, x2, ..., xn | C)
##      pC1 is P(C)
        xtestw1 = np.sum(aw1 * xtest) * pC1
        xtestw2 = np.sum(aw2 * xtest) * pC2
        expected = -1
        if xtestw2 > xtestw1:
            expected = 1
        if expected == Ytest[testIndex]:
            correctCount += 1
        else:
            incorrectCount += 1
    return incorrectCount * 1.0 / (correctCount + incorrectCount)
        
def getFeatureProb(Xtrain, Ytrain, test, j):
    """
    get the probability of a feature j for a given class, known as p(Xj | C)
    test argument is to specify class: -1(first class) vs 1 (second class)
    j argument is to specify feature (or column or word in this case)
    smooth factor k = 1 by default
    """
    k = 1
    Ymatrix = (Ytrain == test).T
    return (np.sum(Xtrain[Ymatrix[0], j]) + k) * 1.0 / (np.sum(Ytrain == test) + k * 2)

def getClassProb(Ytrain, test):
    """
    get the probability of a given class, known as p(C)
    test argument is to specify class: -1(first class) vs 1 (second class)
    """
    classNum = np.sum(Ytrain == test)
    classProb = classNum * 1.0 / Ytrain.size
    return classProb
                
if __name__ == "__main__":
    main()



