import time, math, random, bisect, copy
import datetime as dt
import gym
import Populáció as PN


def geneticAlgorithm(_kornyezet,_mutRata,_populacioSzam, _maxGen,_rejtRetegek,_diszkret  = True,_maxLepesek = 1000,
                                    _info=''):


    kornyezet = gym.make(_kornyezet)

    megfigyeles = kornyezet.reset()
    in_dimen = kornyezet.megfigyeles_space.shape[0]

# Játéktól függően a kezdő paraméterek beállítása folyamatos vagy diszkrét válozókra
    if _diszkret  == False:
        out_dimen = kornyezet.action_space.shape[0]
        actionMin = kornyezet.action_space.low
        actionMax = kornyezet.action_space.high
    else:
        out_dimen = kornyezet.action_space.n
        actionMin = 0
        actionMax = kornyezet.action_space.n

    obsMin = kornyezet.megfigyeles_space.low
    obsMax = kornyezet.megfigyeles_space.high

#  Háló összeállítás: Input réteg -> Paraméter rejtett réteg(ek)
    #                         -> Output réteg
    retegek = []
    retegek.append(in_dimen)
    for i in range(len(_rejtRetegek)):
        retegek.append(_rejtRetegek[i])
    retegek.append(out_dimen)



    populacio = PN.Populacio(_populacioSzam, _mutRata, retegek)  # [in_dimen,13,8,13 ,out_dimen]

    print("\nObservation\n--------------------------------")
    print("Shape :", in_dimen, " | High :", obsMax, " | Low :", obsMin)
    print("\nAction\n--------------------------------")
    print("Shape :", out_dimen, " | High :", actionMax, " | Low :", actionMin, "\n")
#
    maxjutalom = -999999
    generaciok = []
    Stime = dt.datetime.now()

# Algoritmus ciklikus futtatása a generáció max számáig, felírásra kerül a generáció átlagos és legjobb pontszáma
    # és a legjobb generáció is megmarad.
    for generacio in range(_maxGen):
        atlFit = 0.0
        maxFit = -9999

        for egyed in populacio.population:
            teljesJutalom = 0
            for step in range(_maxLepesek):
                # kornyezet.render()
                akcio = egyed.getAkcio(megfigyeles,_diszkret)
                megfigyeles, jutalom, done, info = kornyezet.step(akcio)
                teljesJutalom += jutalom

                if done:
                    megfigyeles = kornyezet.reset()
                    break

            egyed.fitness = teljesJutalom
            atlFit += egyed.fitness
            if egyed.fitness > maxFit:
                maxFit = egyed.fitness
            #Eddigi legjobb eredmény az összes eddigi generációban.
            if maxJutalom < maxFit:
                maxJutalom = maxFit
                
        atlFit /= _populacioSzam
        print("Generation : %3d |  Avg Fitness : %5.0f  |  Max Fitness : %5.0f  | Max Jutalom : %5.0f | Mutation : %5.3f | Layers : " % (generacio + 1, atlFit, maxFit, maxjutalom,_mutRata),_rejtRetegek)
        generaciok.append([generacio + 1, atlFit, maxFit])
        populacio.ujGeneracio()

    Etime = dt.datetime.now()
    elteltIdo = int((Etime-Stime).total_seconds()*1000)
    print(f"-----------The population was evaluated in {elteltIdo} ms-----------")
    # Oszlopok összege: (1. oszlop a generációk száma ami összeadásra kerül ez jelentéktelen adat)
    col_totals = [sum(x) for x in zip(*generaciok)]


    eredmeny = _kornyezet, _mutRata, _rejtRetegek, _maxGen, _populacioSzam, elteltIdo, col_totals, _info, generaciok
    return eredmeny


