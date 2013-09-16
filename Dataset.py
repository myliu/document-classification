import numpy as np
import string

class Dataset:
    """Manages I/O for reading text files with word occurences 
    to produce training and test sets."""
    
    def __createDataMatrices(self, files):
        """Creates complete data matrices from both input datasets."""
        
        X = np.zeros((self.__numExamples, self.__numWords))
        Y = np.zeros((self.__numExamples, 1))
        n = 0

        yc = -1
        for f in files:
            for line in file(f).readlines():
                Y[n] = yc
                words = string.split(line)
                for w in words:
                    w = self.__convertWord(w)
                    idx = self.__wordmap.get(w)
                    if idx is not None and idx >= 0:
                        X[n,idx] = 1
                
                n += 1
            yc += 2

        return (X,Y)
    
    def __convertWord(self, word):
        """Pre-processes words before feature conversion to avoid getting duplicated
        similar words."""
        return word.lower().strip('><');        
        
    def __addFileToMap(self, filename):
        """Adds all the words found in the target file to the word map."""
        
        # build dictionary
        for line in file(filename).readlines():
            words = string.split(line)
            for w in words:
                w = self.__convertWord(w)
                if self.__wordmap.get(w) is not None:
                    self.__wordmap[w] += 1
                else:
                    self.__numWords += 1    
                    self.__wordmap[w] = 1    

            self.__numExamples += 1

    def __cutInfrequentWords(self, cutoff):
        """Cuts infrequent words from the word map, taking only the
        'cutoff' most frequent words."""
        
        self.__numWords = 0
        
        words = self.__wordmap.keys()
        counts = self.__wordmap.values()
        
        idx = np.argsort(counts, None)
        
        self.__wordmap = {}
        for i in idx[-cutoff:]:
            self.__wordmap[words[i]] = self.__numWords
            self.__numWords += 1
        
        print "\tSmallest count after cutoff: ", counts[idx[-cutoff]]
        
    def __init__(self, fileA, fileB, cutoff=2000):
        """Dataset constructor: takes in two filenames for the two classes to test against,
        and also an optional word cutoff (see __cutInfrequentWords)"""

        print "Initializing new Dataset..."

        self.__classSrc = [fileA, fileB]
        self.__wordmap = {}
        self.__numWords = 0
        self.__numExamples = 0
        
        self.__addFileToMap(fileA)
        self.__addFileToMap(fileB)
        
        if cutoff == -1:
            cutoff = self.__numWords
            
        self.__cutInfrequentWords(cutoff)

        print "Initialized dataset:\n\t%s vs. %s\n\t%d words\n\t%d examples" % \
            (fileA, fileB, self.__numWords, self.__numExamples) 
        
        self.__X, self.__Y = self.__createDataMatrices([fileA, fileB])
    
    def getTrainAndTestSets(self, pct, seed=None):
        """Gets a (pct)-(1-pct) random split of the complete dataset and returns
        the result in a 4-tuple (trainX, trainY, testX, testY). Note: pct is in the 
        range [0,1], NOT [0, 100]. Optionally takes in a random seed."""
        
        print "Generating random %g-%g split..." % (100*pct,100*(1-pct))
        
        np.random.seed(seed)
        idx = np.random.permutation(self.__numExamples)

        numTrain = np.floor(self.__numExamples*pct)
        
        trainIdx = idx[0:(numTrain)]
        testIdx = idx[numTrain:]
        
        return (self.__X[trainIdx,:], self.__Y[trainIdx,:], \
                self.__X[testIdx,:], self.__Y[testIdx,:] )
    
    def getWordList(self):
        """Returns the set of words corresponding to the columns of X."""
        
        wordlist = []
        for i in range(0,self.__numWords):
            wordlist.append("")
            
        for (word, id) in self.__wordmap.items():
            if id >= 0:
                wordlist[id] = word
        
        return wordlist

# Sample usage of the Dataset Class    
if __name__ == '__main__':

    d = Dataset("rec.sport.hockey.txt", "rec.sport.baseball.txt", cutoff=2000)
    (Xtrain, Ytrain, Xtest, Ytest) = d.getTrainAndTestSets(0.8, seed=1)
