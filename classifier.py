import numpy as np
import sys

class Classifier():

    def __init__(self, train, test):
        
        self.trainData = np.array(train)
        self.testData = np.array(test)
        
        self.xTrain = self.trainData[:, 1:]
        self.xTest = self.testData[:, 1:]
        
        self.yTrain = self.trainData[:, 0]
        self.yTest = self.testData[:, 0]
        
        self.yes = np.sum(self.yTrain)
        self.no = self.yTrain.size - self.yes

        self.pYes = self.yes/self.yTrain.size
        self.pNo = 1-self.pYes

        self.tables = np.zeros(((self.xTrain[0]).size, 2, 2))
        self.ptests = np.zeros(((self.xTrain[0]).size, 2))

    def tableGen(self, test):
        table = np.zeros((2,2))
        table[0,0] = np.sum(np.logical_and(np.logical_not(self.yTrain), np.logical_not(test)))/self.no
        table[1,0] = np.sum(np.logical_and(np.logical_not(self.yTrain), test))/self.no
        table[0,1] = np.sum(np.logical_and(self.yTrain, np.logical_not(test)))/self.yes
        table[1,1] = np.sum(np.logical_and(self.yTrain, test))/self.yes
        return table
    
    def probGen(self, test):
        passed = np.sum(test)/test.size
        failed = 1 - passed
        return np.array([failed, passed])

    def train(self):
        print("##########")
        print("Starting to Train on ", self.yTrain.size, " data points . . .")
        self.tables = np.apply_along_axis(self.tableGen, 1, self.xTrain.transpose())
        self.pTests = np.apply_along_axis(self.probGen, 1, self.xTrain.transpose()) 
        #print(self.tables)
        print("Training Complete\n")
    
    def predict(self, tests):
        givenYes = 1
        givenNo = 1
        pTests = 1

        for i in range(tests.size):
            givenYes *= self.tables[i][tests[i]][1]
            givenNo *= self.tables[i][tests[i]][0]
            pTests *= self.pTests[i][tests[i]]
        
        #print("Compare1: ",givenYes, givenNo)
        givenYes *= self.pYes
        givenNo *= self.pNo

        #print("Compare2: ",givenYes, givenNo)
        
        givenYes = givenYes/pTests
        givenNo = givenNo/pTests
        
        #print("DivideBy: ",pTests)
        #print("Compare3: ",givenYes, givenNo)

        if(givenYes > givenNo):
            return 1
        else:
            return 0

    def test(self):
        print("Testing on ", self.yTest.size, " data point . . .")
        total = self.yTest.size
        correct = 0
        for i in range(self.yTest.size):
            correct += self.predict(self.xTest[i]) == self.yTest[i]

        accuracy = (float(correct)/float(total))*100
        print("Total Accuracy: = ", accuracy, "%")
        print("##########")

def readFile(fileName):
    lst = []
    with open(fileName) as f:
        for line in f:
            line = line.rstrip("\n").split(",")
            lst.append(list(map(int, line))) 
    return lst


def main():
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    train = readFile(trainFile)
    test = readFile(testFile)
    
    model = Classifier(train, test)  
    model.train()
    model.test()

if __name__ == "__main__":
    main()
        
