import GenetikusAlgoritmus as ga
from multiprocessing import Pool
import GenExcel as ge

# Paraméterek a futtatáshoz
kornyezet = 'LunarLander-v2'#'BipedalWalker-v2' #'CartPole-v1' #'LunarLander-v2'
popSzam = 100
genSzam = 1000
retegek = [13,8,13] # 4,2 es 13,8,13
retegTest = ([13,8,13],[24],[21,9,21],[24,12])
diszkret = True
epizodSzam= 500
mutRatak = [0.001,0.005,0.01,0.05]#0.1,0.250

feladatok = []
# Feladat előkészítse különböző paraméterekkel a többfeladatos futtatáshoz
for i in range(len(mutRatak)):
    p = (kornyezet, mutRatak[i], popSzam, genSzam, retegek, diszkret, epizodSzam)
    feladatok.append(p)

eredmenyek = []

# Feladatok futtatása
if __name__ == '__main__':
    pool = Pool(process=4)
    eredmenyek.append(pool.starmap(ga.GenetikusAlgoritmus, feladatok))
    pool.close()
    pool.join()

    ge.makeExcel(eredmenyek)
# Result: - result[]
#           - result[][] futtatas
#               -result[][][] parameterek
# for i in range(len(MutRatak)):
#     # Paraméterek beállítása egy feladathoz
#     p = (_kornyezet, MutRatak[i], _populacioSzam, _generacioSzam,
#                                             _retegek, _diszkret, _epizodSzam)
#     feladatok.append(p)
#
# pool = Pool(feladatokes=4)
# eredmenyek.append(pool.starmap(ga.genetikusAlgortimus, feladatok))
