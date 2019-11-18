# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:59:58 2019

@author: Lauren Benson
"""


#Want to show: Standard deviation for every connection. So, find std in relation to all repetitions
#for each connection.

import CTRNN_Lauren
import matplotlib.pyplot as plt
import numpy as np

#Global parameters
size = 20
duration = 30
stepsize = 0.1
repetitions = 100


#First, show std for 0 connection.


TotalAvg = 0
EachRepOutputList = []
for r in range(repetitions):
    time = np.arange(0.0,duration,stepsize)
    nn = CTRNN_Lauren.CTRNN(size)
    nn.randomizeParameters()
    nn.setweightmatrix3(0)
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
        TimeSum = TimeSum + SumNewOutput
        step = step+1
    NewTimeSum = (TimeSum) / (duration * size)
    EachRepOutputList.append(NewTimeSum)
    



plt.plot(time, EachRepOutputList)
#plt.errorbar(possible_connections, NewTotalAvgList, yerr=Errors, fmt='o')
plt.xlabel("Number of connections")
plt.ylabel("Change in Neural Output")
plt.title("Neural activity")
plt.show()        
        

