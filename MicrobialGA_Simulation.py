# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:13:40 2019

@author: benso
"""

import mga
import numpy as np


population_size = 20
genotype = 10
#total_pop = np.zeros((population_size, genotype))
transfectprob = 0.5
mutationprob = 0.5
generations = 5
num_tournaments = generations * population_size

def fitness_func(genotype):
    fitness_scores = np.sum(genotype)
    return fitness_scores
ga = mga.Microbial(fitness_func, population_size, genotype, transfectprob, mutationprob)
ga.run(num_tournaments)
