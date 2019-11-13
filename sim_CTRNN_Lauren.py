# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:21:02 2019

@author: Lauren Benson
"""

import CTRNN_Lauren
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 
# Things I did since 11/12:
#     Correctly ordered connections (0-size^2)
#     Error bars on figure
#     
# =============================================================================

#Global parameters
size = 50
duration = 50
stepsize = 0.1
repetitions = 100

NewTotalAvgListStd = []
NewTotalAvgList = []
possible_connections = np.arange(0,size**2,1)
possible_connections_ordered = np.flip(possible_connections)
for c in possible_connections_ordered:
    TotalAvg = 0
    for r in range(repetitions):
        time = np.arange(0.0,duration,stepsize)
        nn = CTRNN_Lauren.CTRNN(size)
        nn.randomizeParameters()
        nn.setweightmatrix3(c)
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
        TotalAvg = NewTimeSum + TotalAvg
    NewTotalAvg = TotalAvg / repetitions #this is the average neural output over all reps (one value)
    NewTotalAvgList.append(NewTotalAvg) #This is an array of each average neural output for each number of connections
    NewTotalAvgListStd.append(np.std(NewTotalAvgList))
    Errors = NewTotalAvgListStd/(np.sqrt(len(NewTotalAvgList)))
    


plt.plot(possible_connections, NewTotalAvgList)
plt.errorbar(possible_connections,NewTotalAvgList,yerr=Errors,fmt='o')
plt.xlabel("Number of connections")
plt.ylabel("Output")
plt.title("Neural activity")
plt.show()        
        