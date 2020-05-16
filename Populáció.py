import numpy as np
import random, bisect, copy

#
## Aktiváló függvények
#
def reLu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
#
def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x))
    return x


# Neurális háló inicializálása
class NeuralisHalo:
    def __init__(self, _retegTomb, _aktivacio = False):
        self.retegTomb = _retegTomb
        self.sulyok = []
        self.eltolassulyok = []
        self.fitnesz = 0
        self.aktivacio = _aktivacio
        for i in range(len(self.retegTomb) - 1):
            ##! Súlyok beállítása, a háló rétegei random értékekkel töltődnek fel, méretei a megadott neuronszám alapján jönnek létre
            self.sulyok.append(np.random.uniform(low=-1, high=1, size=(self.retegTomb[i], self.retegTomb[i + 1])).tolist())
            self.eltolassulyok.append(np.random.uniform(low=-1, high=1, size=(self.retegTomb[i + 1])).tolist())

    def getAkcio(self,bemenet,_diszkret):
        kimenet = bemenet
        for i in range(len(self.retegTomb) - 1):
            retegkimenet = np.dot(kimenet, self.sulyok[i]) + self.eltolassulyok[i]
            kimenet = np.reshape(retegkimenet, (self.neuronSzam[i + 1]))

            if(self.aktivacio): kimenet = sigmoid(kimenet)
        return np.argmax(kimenet) if _diszkret else kimenet

    # Populáció inicializálása, ezen belül inicializalodik a neurális háló
    class Populáció:
        def __init__(self, populacioSzam, retegTomb, mutSzam):
            self.retegTomb = retegTomb
            self.populacioSzam = populacioSzam
            self.mutRata = mutSzam
            # A populáció feltöltése egyedekkel
            self.populacio = [ NeuralisHalo(retegTomb) for _ in range(populacioSzam)]

            # Súyok mutációja és rekombinációja
            def ujEgyed(self, szulo1, szulo2):
                egyed = NeuralisHalo(self.retegTomb)
                for i in range(len(egyed.sulyok)):
                    for j in range(len(egyed.sulyok[i])):
                        for k in range(len(egyed.sulyok[i][j])):
                            if random.random() > self.mutRata:  # Amennyiben kisebb mint a mut_rate marad a random weight
                                if random.random() < szulo1.fitnesz / (
                                        szulo1.fitnesz + szulo2.fitnesz):  # Ha az első szülő ügyesebb,több esélye van átadni a genetikáját
                                    egyed.sulyok[i][j][k] = szulo1.sulyok[i][j][k]
                                else:
                                    egyed.sulyok[i][j][k] = szulo2.sulyok[i][j][k]
                # Eltolássúlyok mutációja, és rekombinációja
                for i in range(len(egyed.eltolassulyok)):
                    for j in range(len(egyed.eltolassulyok[i])):
                        if random.random() > self.mutRata:
                            if random.random() < szulo1.fitnesz / (szulo1.fitnesz + szulo2.fitnesz):
                                egyed.eltolassulyok[i][j] = szulo1.eltolassulyok[i][j]
                            else:
                                egyed.eltolassulyok[i][j] = szulo2.eltolassulyok[i][j]
                return egyed

            def ujGeneracio(self):
                ujGen = []
                # A populációt visszafelé rendezzük fitnesz szerint, a legnagyobb értékű az első elem
                self.populacio.sort(key=lambda x: x.fitnesz, reverse=True)
                for i in range(self.populacioSzam):
                    if random.random() < float(self.populacioSzam - i) / self.populacioSzam:
                        ujGen.append(copy.deepcopy(self.populacio[i]))
                #Keresztezés az első két egyedből
                while (len(ujGen) < self.populacioSzam):
                    ujGen.append(self.ujEgyed(ujGen[0], ujGen[1]))
                self.populacio = ujGen

