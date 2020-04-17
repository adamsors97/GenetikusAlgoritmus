import GenetikusAlgoritmus as ga
from multiprocessing import Pool
import GenExcell as GE

# Paraméterek a futtatáshoz
GAME = 'LunarLander-v2'#'BipedalWalker-v2' #'CartPole-v1' #'LunarLander-v2'
PopCount = 10
GenCount = 1
Layers = [13,8,13] # 4,2 es 13,8,13
LayerTest = ([13,8,13],[24],[21,9,21],[24,12])
Discrete = True
EpCount= 500
Replay = False
Decay = False
MutRates = [0.001,0.005,0.01,0.05]#0.1,0.250
process = []

# Feladat előkészítse különböző paraméterekkel a többfeladatos futtatáshoz
for i in range(len(MutRates)):
    p = (GAME, MutRates[i], PopCount, GenCount, Layers, Discrete, EpCount, Replay, Decay)
    process.append(p)

results = []

# Feladatok futtatása
if __name__ == '__main__':
    pool = Pool(processes=4)
    results.append(pool.starmap(ga.geneticAlgorithm, process))
    pool.close()
    pool.join()
    print(results[0][1][1])
    print('1.dim: ', len(results),', 2.dim: ',len(results[0]),' 3.dim: ',len(results[0][0]))
    GE.makeExcel(results)
# Result: - result[]
#           - result[][] futtatas
#               -result[][][] parameterek
