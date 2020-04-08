import time, math, random, bisect, copy
import datetime as dt
import gym
import numpy as np
import Population as PN
import GenExcell as GE



def reLu(x):
    return np.maximum(0, x)
def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x))
    return x
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def geneticAlgorithm(Game,Mut_Rate,Pop_Count,Max_Gen,Hid_Layers,Discrete = True,Max_Steps = 1000,
                                    replayBots=False,_decay = False,extraString='', _fitness = True):

    MAX_STEPS = Max_Steps
    MAX_GENERATIONS = Max_Gen
    POPULATION_COUNT = Pop_Count
    MUTATION_RATE = Mut_Rate #0.005  # 0.001
    DECAY_RATE = Mut_Rate
    env = gym.make(Game)

    observation = env.reset()
    in_dimen = env.observation_space.shape[0]

# Játéktól függően a kezdő paraméterek beállítása folyamatos vagy diszkrét válozókra
    if Discrete == False:
        out_dimen = env.action_space.shape[0]
        actionMin = env.action_space.low
        actionMax = env.action_space.high
    else:
        out_dimen = env.action_space.n
        actionMin = 0
        actionMax = env.action_space.n

    obsMin = env.observation_space.low
    obsMax = env.observation_space.high

#  Háló összeállítás: Input réteg -> Paraméter réteg(ek) -> Output réteg
    layers = []
    layers.append(in_dimen)
    for i in range(len(Hid_Layers)):
        layers.append(Hid_Layers[i])
    layers.append(out_dimen)

    pop = PN.Population(POPULATION_COUNT, MUTATION_RATE, layers)  # [in_dimen,13,8,13 ,out_dimen]
    bestNeuralNets = []

    print("\nObservation\n--------------------------------")
    print("Shape :", in_dimen, " | High :", obsMax, " | Low :", obsMin)
    print("\nAction\n--------------------------------")
    print("Shape :", out_dimen, " | High :", actionMax, " | Low :", actionMin, "\n")
#
    maxreward = -999999
    generations = []
    Stime = dt.datetime.now()
# Algoritmus ciklikus futtatása a generáció max számáig, felírásra kerül a generáció átlagos és legjobb pontszáma
    # és a legjobb generáció is megmarad.
    for gen in range(MAX_GENERATIONS):
        genAvgFit = 0.0
        maxFit = -100000000
        maxNeuralNet = None
        for nn in pop.population:
            totalReward = 0
            for step in range(MAX_STEPS):
                # env.render()
                action = nn.getOutput(observation,Discrete)
                observation, reward, done, info = env.step(action)
                totalReward += reward

                if done:
                    observation = env.reset()
                    break

            nn.fitness = totalReward
            genAvgFit += nn.fitness
            if nn.fitness > maxFit:
                maxFit = nn.fitness
                maxNeuralNet = copy.deepcopy(nn);
            #Eddigi legjobb eredmény az összes eddigi generációban.
            if maxreward < maxFit:
                maxreward = maxFit

        bestNeuralNets.append(maxNeuralNet)
        genAvgFit /= pop.popCount
        # if _decay:
        #     if maxreward*0.35 > genAvgFit :
        #         DECAY_RATE = min(0.05,DECAY_RATE + 0.005)
        #         MUTATION_RATE = min(0.25, MUTATION_RATE + DECAY_RATE)
        #     elif maxreward*0.6 > genAvgFit :
        #         DECAY_RATE = min(0.05,DECAY_RATE - 0.001)
        #         MUTATION_RATE = min(0.25, MUTATION_RATE + DECAY_RATE)
        #     elif genAvgFit > maxreward*0.75 :
        #         DECAY_RATE = max(-0.05,DECAY_RATE - 0.01)
        #         MUTATION_RATE = max(0, min(0.25,MUTATION_RATE + DECAY_RATE))
        print("Generation : %3d |  Avg Fitness : %5.0f  |  Max Fitness : %5.0f  | Max Reward : %5.0f | Decay : %5.3f | Mutation : %5.3f | Layers : " % (gen + 1, genAvgFit, maxFit, maxreward,DECAY_RATE,MUTATION_RATE),Hid_Layers)
        generations.append([gen + 1, genAvgFit, maxFit])
        pop.createNewGeneration(maxNeuralNet)

    Etime = dt.datetime.now()
    elapsedtime = int((Etime-Stime).total_seconds()*1000)
    print(f"-----------The population was evaluated in {elapsedtime} ms-----------")
    col_totals = [sum(x) for x in zip(*generations)]

    def replayBestBots(bestNeuralNets, steps, sleep):
        for i in range(len(bestNeuralNets)):
            if i % steps == 0:
                observation = env.reset()
                print("Generation %3d had a best fitness of %4d" % (i, bestNeuralNets[i].fitness))
                for step in range(MAX_STEPS):
                    env.render()
                    #time.sleep(sleep)
                    action = bestNeuralNets[i].getOutput(observation,Discrete)
                    bestNeuralNets[i].printWeightsandBiases()
                    observation, reward, done, info = env.step(action)
                    if done:
                        observation = env.reset()
                        break
                print("Steps taken =", step)

    # GE.makeExcel(generations,MUTATION_RATE,Game,MAX_GENERATIONS,POPULATION_COUNT,elapsedtime,col_totals,extraString)
    if replayBots == True:
        print("Replaying best performing agents of their generation :")
        replayBestBots(bestNeuralNets, max(1, int(math.ceil(MAX_GENERATIONS / 10.0))), 0.0625)

    return generations