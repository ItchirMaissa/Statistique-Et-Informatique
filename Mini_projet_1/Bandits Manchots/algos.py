#!/usr/bin/env python
# coding: utf-8

# In[60]:


#import random
# L'algorithme aleatoir est une methode simple pour choisir une action parmi un ensemble d'actions possibles 

#def algorithme_aleatoir(actions):
    #return random.choice(actions)


#actions_possibles = [1, 2, 3, 4, 5]
#action_choisie = algorithme_aleatoir(actions_possibles)
#print("Action choisie:", action_choisie)


# In[89]:


from math import log, sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

def effectuer_tirage(leviers, a_t):
    if a_t < 0 or a_t >  len(leviers):
        print("L'action n'est pas valide")
        return
    action_choisie = leviers[a_t]
    print(action_choisie)
    if random.random() < action_choisie:
        return 1  # succes
    else:
        return 0  # echec 
    
def tirage_levier(i):
    data = pd.read_csv('probabilities1.csv')
    leviers = data['Probabilities'].tolist()
    return effectuer_tirage(leviers, i)



        
        
# a_t  Est l'indice de l'action  choisie
# U_t Une liste des récompenses moyennes estimées pour chaque action  
# N_t Le nombre de fois où chaque action a été choisie
# r_t la récompense obtenue lorsque le joueur joue à l'instant 


# In[90]:


# Une fonction pour mettre à jour 
def mettre_à_jour(U_t, N_t, a_t, r_t):
    if a_t < 0 or a_t >= len(U_t):
        print("L'action n'est pas valide")
        return 
    U_t[a_t] = (U_t[a_t] * N_t[a_t] + r_t ) / ( N_t[a_t] + 1 )
    N_t[a_t] += 1
    


# In[91]:


def algo_aleatoire(U_t, N_t):
    i = random.randint(0,len(U_t)-1)
    result = tirage_levier(i)
    mettre_à_jour(U_t=U_t, N_t=N_t, a_t=i, r_t=result)
    return result


# In[92]:


def algo_greedy(U_t, N_t):
    # Au debut,  l'algorithme choisit chaque levier de manière uniforme, sans préférence particulière.
    # Au debut; On appelle la fonction algo_aleatoire 
    # On decide de consacrer un certin nombre de tours à l'exploration 
    debut = 20
    if sum(N_t)  < debut:# je comprends pas cette condition
        return algo_aleatoire(U_t, N_t) #  Phase d'exploration
    # On trouve l'index du levier avec la récompense estimée la plus élevée
    i = U_t.index(max(U_t))
    result = tirage_levier(i)
    mettre_à_jour(U_t=U_t, N_t=N_t, a_t=i, r_t=result)
    return result 


# In[151]:


def algo_epsilon_greedy(U_t, N_t):
    debut = 20
    alea = random.random()
    epsilon = 0.9
    if sum(N_t)  < debut or alea < epsilon:
        return algo_greedy(U_t, N_t)
    i = U_t.index(max(U_t))
    result = tirage_levier(i)
    mettre_à_jour(U_t=U_t, N_t=N_t, a_t=i, r_t=result)
    return result


# In[94]:


def getArgMax(U_t,N_t,t):
    ma_fonction = []
    N = len(U_t)
    for i in range(0,N):
        if N_t[i] == 0:
            ma_fonction.append(0)
        else:
            ma_fonction.append(U_t[i] + sqrt( 2 * log(t) / N_t[i]))
    return ma_fonction.index(max(ma_fonction))


# In[95]:



def algo_UCB(U_t,N_t):
    first = 20
    t = sum(N_t)
    if t < first :
        return algo_aleatoire(U_t, N_t)
    i = getArgMax(U_t, N_t, t)
    result = tirage_levier(i)
    mettre_à_jour(U_t=U_t, N_t=N_t, a_t=i, r_t=result)
    return result


# In[163]:


def apply_algos(N, T, num_algo):
    if num_algo not in [1,2,3,4]:
        return
    N_t = [0] * N 
    U_t = [0] * N 
    for t in range(0,T):
        if num_algo == 1:
            algo_aleatoire(U_t, N_t)
        elif num_algo == 2:
            algo_greedy(U_t, N_t)
        elif num_algo == 3:
            algo_epsilon_greedy(U_t, N_t)
        elif num_algo == 4:
            algo_UCB(U_t, N_t)
    return U_t

def get_decision(N,T,num_algo):
    U_hat = apply_algos(N,T,num_algo)
    return U_hat.index(max(U_hat))

def test(N,T):
    length = 800 
    decisions = [0]*5
    result = np.zeros((5,length),dtype=np.int16)

    for j in range(length):
        for num_algo in range(1,5):
            
            decisions[num_algo] = get_decision(N,T,num_algo)
            result[num_algo,j] = tirage_levier(decisions[num_algo])
        
        print(".", end="")
        if j % 20 == 0:
            print(j,"times")
        print()
    # afficher le résultat
    print(result.sum(axis=1)[1:4])

def regret_algos(N, T, num_algo):
   
    if num_algo not in [1,2,3,4]:
        return
    N_t = [0] * N 
    U_t = [0] * N 
    r = [0] * T
    regret_function = [0] * T
    for t in range(T):
        if num_algo == 1:
            r[t] = algo_aleatoire(U_t, N_t)
        elif num_algo == 2:
            r[t] = algo_greedy(U_t, N_t)
        elif num_algo == 3:
            r[t] = algo_epsilon_greedy(U_t, N_t)
        elif num_algo == 4:
            r[t] = algo_UCB(U_t, N_t)
        regret_function[t] = t * max(U_t) - sum(r)
    return regret_function

def draw_graph(N,T):
    
    pyplot.plot(np.arange(T),regret_algos(N,T,1),label="random",color='r')    
    pyplot.plot(np.arange(T),regret_algos(N,T,2),label="Greedy",color='b')
    pyplot.plot(np.arange(T),regret_algos(N,T,3),label="Epsilon-greedy",color='y')
    pyplot.plot(np.arange(T),regret_algos(N,T,4),label="UCB",color='g')
    pyplot.xlabel('T')
    pyplot.ylabel('regret')
    pyplot.legend(loc="upper left")
    pyplot.show()

def main():
    N = 10
    T = 100
    
    
    draw_graph(N,T)

if __name__ == "__main__":
    main()
    







