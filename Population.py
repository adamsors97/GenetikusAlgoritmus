import numpy as np
import random, bisect, copy

#
## Aktiváló funkciók
#
def reLu(x):
    return np.maximum(0, x)
def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x))
    return x
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
#


# Neurális háló inicializálása
class NeuralNet:
    def __init__(self, nodeCount):
        self.fitness = 0.0
        self.nodeCount = nodeCount
        self.weights = []
        self.biases = []
        for i in range(len(nodeCount) - 1):
            ##! Súlyok beállítása, a háló rétegei random értékekkel töltődnek fel, méretei a megadott node szám alapján jönnek létre
            self.weights.append(np.random.uniform(low=-1, high=1, size=(nodeCount[i], nodeCount[i + 1])).tolist())
            self.biases.append(np.random.uniform(low=-1, high=1, size=(nodeCount[i + 1])).tolist())

    ## Akció lekérése a súlyok alapján, aktivációs funkció használata
    def getOutput(self, input, Discrete):
        output = input
        # output = input / max(np.max(np.linalg.norm(input)), 1)
        for i in range(len(self.nodeCount) - 1):
            output = np.reshape(np.matmul(output, self.weights[i]) + self.biases[i],
                                (self.nodeCount[i + 1]))  # tanh(np.matmul)
            output = sigmoid(output)
        return np.argmax(output) if Discrete else output  # np.argmax(output) egy elemu / output tomb # softmax

# Populáció inicializálása, ezen belül jön létre a neurális háló
class Population:
    def __init__(self, populationCount, mutationRate, nodeCount):
        self.nodeCount = nodeCount
        self.popCount = populationCount
        self.m_rate = mutationRate
        self.population = [NeuralNet(nodeCount) for _ in range(populationCount)]

## Leszármaztatás, mutációs ráta alapján mutálás
    def createChild(self, nn1, nn2):
        child = NeuralNet(self.nodeCount)
        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    if random.random() > self.m_rate:  # Amennyiben kisebb mint a mut_rate marad a random weight
                        if random.random() < nn1.fitness / (
                                nn1.fitness + nn2.fitness):  # Ha az első szülő ügyesebb,több esélye van átadni a genetikáját
                            child.weights[i][j][k] = nn1.weights[i][j][k]
                        else:
                            child.weights[i][j][k] = nn2.weights[i][j][k]

        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                if random.random() > self.m_rate:
                    if random.random() < nn1.fitness / (nn1.fitness + nn2.fitness):
                        child.biases[i][j] = nn1.biases[i][j]
                    else:
                        child.biases[i][j] = nn2.biases[i][j]

        return child

    ## Generáció szelekció, crossover, létrehozás
    def createNewGeneration(self, bestNN):
        nextGen = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.popCount):
            if random.random() < float(self.popCount - i) / self.popCount:
                nextGen.append(copy.deepcopy(self.population[i]))

        while (len(nextGen) < self.popCount):
            nextGen.append(self.createChild(nextGen[0], nextGen[1]))

        self.population.clear()
        self.population = nextGen