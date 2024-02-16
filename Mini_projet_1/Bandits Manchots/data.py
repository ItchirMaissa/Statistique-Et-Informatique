#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

# Générer un échantillon de 700 nombres selon une distribution uniforme
probs = np.random.uniform(low=0, high=0.9, size=10)

# Créer un DataFrame
data = pd.DataFrame({'Probabilities': probs})

# Enregistrer le DataFrame dans un fichier CSV
data.to_csv('probabilities2.csv', index=False)





