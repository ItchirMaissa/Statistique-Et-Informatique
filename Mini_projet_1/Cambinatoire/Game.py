#!/usr/bin/env python
# coding: utf-8


import random
import numpy as np
import matplotlib.pyplot as plt

class Player:
    def __init__(self, signal=1, name="untitled"):
        self.signal = signal # 1 ou -1
        self.name = name
        self.nbStep = 0 # nombre de coup

    def getOneStep(self, game):
        # print(self.name, "has finished one step")
        return random.choice(game.get_available_columns())

    def reset(self):
        self.nbStep = 0





import numpy as np

class Game:

    def __init__(self, num_line, num_column):
        self.nRow = num_line
        self.nCol = num_column
        self.myBoard = np.zeros((num_line, num_column), dtype=int)
        self.winBoard = self.get_win_board()

    def is_complete(self):
        for line in self.myBoard:
            for item in line:
                if item == 0:
                    return False
        return True

    def reset(self, players):
        self.myBoard = np.zeros((self.nRow, self.nCol), dtype=int)
        for player in players:
            player.reset()

    def get_win_board(self):
        result = np.empty((0, 4, 2), dtype=int)
        for i in range(0, self.nRow):
            for j in range(0, self.nCol):
                if i + 3 < self.nRow:
                    temp = np.array([[(i, j), (i + 1, j), (i + 2, j), (i + 3, j)]])
                    result = np.append(result, temp, axis=0)
                if j + 3 < self.nCol:
                    temp = np.array([[(i, j), (i, j + 1), (i, j + 2), (i, j + 3)]])
                    result = np.append(result, temp, axis=0)
                if i + 3 < self.nRow and j + 3 < self.nCol:
                    temp = np.array([[(i, j), (i + 1, j + 1), (i + 2, j + 2), (i + 3, j + 3)]])
                    result = np.append(result, temp, axis=0)
                if i + 3 < self.nRow and j - 3 >= 0:
                    temp = np.array([[(i, j), (i + 1, j - 1), (i + 2, j - 2), (i + 3, j - 3)]])
                    result = np.append(result, temp, axis=0)
        return result

    def has_won(self):
        for line in self.winBoard:
            cnt = 0
            for point in line:
                cnt += self.myBoard[point[0], point[1]]
            if np.abs(cnt) == 4:
                #  gagné si qutre -1 ou qutre 1, sinon personne ne gagner
                return True
        return False

    def play(self, x, player: Player):
        #  verifier la validation du nombre de colone
        if self.is_complete() or x < 0 or x >= self.nCol:
            return False
        # commencer par la première ligne
        i = 0
        #  parcourir ver la ligne nrow
        while i < self.nRow and self.myBoard[i][x] == 0:
            # si l'on arrive à la fin, ou rencontre un non-zéro (déjà existe une pièce), alore on sort le boucle
            i += 1
        # déplacer dans la dernière place où il y a une pièce
        i -= 1
        if i == -1:
            # si toute la colone est plein, c-à-d le while est jamais exécuté, alors il y a un problème
            return False
        #  placer une pièce ici
        self.myBoard[i, x] = player.signal

        return True
    
    def get_available_columns(self):
        columns = []
        for i in range(self.nCol):
            if self.myBoard[0][i] == 0:
                columns.append(i)
        return columns

    def is_finished(self):
        return self.is_complete() or self.has_won()

    def run(self, *players: Player, showChessBoard=True, showResult=True, restart=True):
        if restart:
            self.reset(players)
        while True:
            for player in players:
                #  faire une action
                self.play(player.getOneStep(self), player)        
                #  et ajouter un coup
                player.nbStep += 1
                # afficher la plateau
                if showChessBoard:
                    print(self.myBoard)
                #  la jury du résultat
                if self.has_won():
                    if showResult:
                        print(player.name, "won")
                    # retourner le gagnante
                    return player
                elif self.is_complete():
                    if showResult:
                        print("END")
                    return 0





class RandomPlayer(Player):

    def __init__(self, signal=1, name="untitled"):
        super().__init__(signal, name)

    def getOneStep(self, game):
        # print(self.name,"has finished one step")
        return random.choice(game.get_available_columns())




import copy
import random

class MonteCarloPlayer(Player):

    def __init__(self, signal=1, name="untitled", simulation_times=50):
        super().__init__(signal, name)
        self.num_sim = simulation_times


    def division_array(self,array_mem, array_denom):
        result = []
        length = min(len(array_mem), len(array_denom))
        for i in range(length):
            if array_denom[i] == 0:
                if array_mem[i] == 0:
                    result.append(0)
                else:
                    result.append(float('inf'))
            else:
                result.append(array_mem[i] / array_denom[i])
        return result

    def getOneStep(self, game):
        # print("Monte")
        gain = [0] * game.nCol
        total = [0] * game.nCol
        available_columns = game.get_available_columns()
        n = self.num_sim
        player1 = RandomPlayer(signal=-self.signal)
        player2 = RandomPlayer(signal=self.signal)
        for i in range(n):
            # faire un copy de jeu
            game_copy = copy.deepcopy(game)
            #  choisir une colonne au hasard
            x = random.choice(available_columns)
            # placer une pièce d'abord dans la colonne choisie
            game_copy.play(x, self)
            # si l'on a gagné déjà, pas besion de continuer
            if game_copy.has_won():
                return x
            #  On y va !!! 
            result = game_copy.run(player1, player2, showChessBoard=False, showResult=False, restart=False)
            if not result == 0: # si c'est pas une partie nulle
                #  incrémenter le nombre de fois total
                total[x] += 1 
                if result.signal == self.signal: # si c'est soi-même qui gagne
                    gain[x] += 1
        if sum(gain) == 0: #  Si jamais gagner
            # choisir au hazard
            return random.choice(available_columns)
        #  probs <- gain / total
        probs = self.division_array(gain,total)
     
        # retourner l'indece de max, ça vaut dire le colonne où la prba est la plus grande
        return probs.index(max(probs))





from math import sqrt,log
import random
import copy

class UCTPlayer(Player):
  
    def __init__(self, signal=1, name="untitled", simulation_times=50, first=10):
        super().__init__(signal, name)
        self.num_sim = simulation_times
        self.first = first
    
    def division_array(self,array_mem, array_denom):
        result = []
        length = min(len(array_mem), len(array_denom))
        for i in range(length):
            if array_denom[i] == 0:
                if array_mem[i] == 0:
                    result.append(0)
                else:
                    result.append(float('inf'))
            else:
                result.append(array_mem[i] / array_denom[i])
        return result
    
    def getArgMax(self,mu_t,N_t,t):
        my_func = []
        N = len(mu_t)
        for i in range(0,N):
            if N_t[i] == 0:
                my_func.append(0)
            else:
                my_func.append(mu_t[i] + sqrt( 2 * log(t) / N_t[i]))
        return my_func.index(max(my_func))

    def getOneStep(self, game):
        # print("UCT")
        #  initialiser les longeurs
        N = game.nCol
        T = self.num_sim
        #  initialiser les variable intermédiaire
        a = []
        # N_t (i)
        N_t = [0] * N
        # mu_t (i)
        mu_t = [0] * N
        # r(t)
        r = [0] * T
        gain = [0] * game.nCol
        total = [0] * game.nCol
        # obtenir les colonnes valables
        available_columns = game.get_available_columns()
        #  initialiser deux joueurs
        player1 = RandomPlayer(signal=-self.signal)
        player2 = RandomPlayer(signal=self.signal)
        #  commencer le proccess d'itération
        for t in range(T):
            #  faire un copy de jeu
            game_copy = copy.deepcopy(game)
            if t < self.first :
                # choisir une colonne au hasard, et mettre dans a[t]
                a.append(random.choice(available_columns))
            else:
                a.append(self.getArgMax(mu_t, N_t, t-1))
            # mettre le résultat dans N_t
            N_t[a[t]] += 1
            # placer une pièce d'abord dans la colonne choisie
            game_copy.play(a[t], self)
            # si l'on a gagné déjà, pas besion de continuer
            if game_copy.has_won():
                return a[t]
            # On y va !!! 
            result = game_copy.run(player1, player2, showChessBoard=False, showResult=False, restart=False)
            # jury
            if not result == 0: #  si c'est pas une partie nulle
                #  incrémenter le nombre de fois total
                total[a[t]] += 1 
                if result.signal == self.signal: #  si c'est soi-même qui gagne
                    gain[a[t]] += 1
            #  calcule mu_t
            if total[a[t]] != 0:
                mu_t[a[t]] = gain[a[t]] / total[a[t]]

        if sum(gain) == 0: #  Si jamais gagner
            #  choisir au hazard
            return random.choice(available_columns)
        # retourner l'indece de max, ça vaut dire le colonne où la prba est la plus grande
        return mu_t.index(max(mu_t))





import matplotlib.pyplot as ply

BOARD_LENGTH = 7
BOARD_HEIGHT = 6
PLAYER1 = 1
PLAYER2 = -1

def count_and_analyze(player1,player2,num_round=100):
    oneGame = Game(BOARD_HEIGHT, BOARD_LENGTH)
    nb_coups_max = BOARD_HEIGHT*BOARD_LENGTH//2+2
    countWin_player1 = [0]*(nb_coups_max)
    countWin_player2 = [0]*(nb_coups_max)
    player1_win = 0
    player2_win = 0
    nb_tie = 0 #nobody win the game

    for i in range(0,num_round):
        if i%2 == 0:
            result = oneGame.run(player1,player2,showChessBoard=False)
        else:
            result = oneGame.run(player2,player1,showChessBoard=False)
        if not result == 0 :
            if result.signal == player1.signal:
                countWin_player1[result.nbStep] += 1
                player1_win += 1
            elif result.signal == player2.signal:
                countWin_player2[result.nbStep] += 1
                player2_win += 1
        else:
            nb_tie += 1
            
    print("countWin_player1:", countWin_player1)
    print("countWin_player2:", countWin_player2)    
        

     


    print(player1.name,"won",countWin_player1,"in total",player1_win,"times")
    print(player2.name,"won",countWin_player2,"in total",player2_win,"times")
    print("there are ",nb_tie,"ties during the games, and the probability is",nb_tie/num_round) #times of tie

    proba1 = np.array(countWin_player1)/num_round
    proba2 = np.array(countWin_player2)/num_round
    
    print("proba1:", proba1)
    print("proba2:", proba2)


    plt.bar(np.arange(nb_coups_max)-0.2,proba1,width=0.4,label=player1.name,color='r')
    plt.bar(np.arange(nb_coups_max)+0.2,proba2,width=0.4,label=player2.name,color='b')
    plt.xlabel('nb_de_coups')
    plt.ylabel('probabilite')
    plt.legend(loc="upper left")
    plt.show()

def part1():
    #print(np.size(oneGame.get_win_board(),0))
    #on a une liste de 69 quadruplets de cases
    player1 = RandomPlayer(signal=PLAYER1, name="Random1")
    player2 = RandomPlayer(signal=PLAYER2, name="Random2")
    count_and_analyze(player1,player2,1000)
    
def part2():
    #MonteCarlo VS Random:
    player1 = MonteCarloPlayer(signal=PLAYER1, name="MonteCarlo")
    player2 = RandomPlayer(signal=PLAYER2, name="Random")
    count_and_analyze(player1,player2,100)
    #MontreCarlo VS MontreCarlo:
    player1 = MonteCarloPlayer(signal=PLAYER1, name="MonteCarlo1")
    player2 = MonteCarloPlayer(signal=PLAYER2, name="MonteCarlo2")
    count_and_analyze(player1,player2,100)
    
def part3():
    # MonteCarlo VS UCTPlayer:
    player1 = MonteCarloPlayer(signal=PLAYER1, name="MonteCarlo", simulation_times=100)
    player2 = UCTPlayer(signal=PLAYER2, name="UCT",  simulation_times=100 ,first=20)
    # count_and_analyze(player1,player2,100)
    oneGame = Game(BOARD_HEIGHT, BOARD_LENGTH)
    oneGame.run(player1,player2)

def part4():
    # Random VS UCTPlayer:
    player1 = RandomPlayer(signal=PLAYER1, name="Random")
    player2 = UCTPlayer(signal=PLAYER2, name="UCT",  simulation_times=100 ,first=20)
    count_and_analyze(player1,player2,100)
    
if __name__ == "__main__":
    part1()
    #part2()
    #part3()
     #part4()
    
    
    print("The program finished")






