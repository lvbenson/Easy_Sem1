# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:35:42 2019

@author: Lauren Benson

"""

import CTRNN_Lauren
import matplotlib.pyplot as plt
import numpy as np


size = 100
duration = 50
stepsize = 0.1

#Manually change number of connections each time to see plot 

NewRepOutputList = []
repetitions = np.arange(0,100,1)
TotalAvg = 0
EachRepOutputList = []
Std_Each_Rep = []
for r in repetitions:
    time = np.arange(0.0,duration,stepsize)
    nn = CTRNN_Lauren.CTRNN(size)
    nn.randomizeParameters()
    nn.setweightmatrix3(1000)
    nn.initializeState(np.zeros(size))
    outputs1 = np.zeros((len(time)+1,size))
    outputs1[0] = nn.Output
    TimeSum = 0
    step = 1
    for t in time:
        nn.step(stepsize)
        outputs1[step] = nn.Output
        NewOutput = np.absolute(outputs1[step] - outputs1[step-1])
        SumNewOutput = np.sum(NewOutput)
        TimeSum = TimeSum + SumNewOutput #sum of change in outputs
        step = step+1
        NewTimeSum = (TimeSum) / (duration * size) 
    EachRepOutputList.append(NewTimeSum) #This is calculating the change in neural output for every repetition
    Std_Each_Rep.append(np.std(EachRepOutputList)) #the standard deviation for each set of repetitions 
    Errors = Std_Each_Rep/(np.sqrt(len(EachRepOutputList))) #error bars for each set of repetitions 
    np.save('Change in neural output for every repetition',EachRepOutputList)
    

plt.plot(repetitions, EachRepOutputList)
plt.errorbar(repetitions,EachRepOutputList,yerr=Errors, label = 'neural activity for all repetitions', fmt='o')
plt.xlabel("Reptitions")
plt.ylabel("Output")
plt.title("Neural activity")
plt.legend(loc='lower right')
plt.show()        
        