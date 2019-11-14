# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:32:43 2019

@author: Lauren Benson
"""

import CTRNN_Lauren
import matplotlib.pyplot as plt
import numpy as np

#Global parameters
size = 3
duration = 30
stepsize = 0.1
repetitions = 100

#What this does: Changes entire time constant vector to one specified value.
#Need to do: Have different values for time constants? Each neuron needs different t.c.?


NewTotalAvgList = []
Possible_Time_Constants = np.arange(0.1,5,0.1)
for t in Possible_Time_Constants:
    TotalAvg = 0
    for r in range(repetitions):
        time = np.arange(0.0,duration,stepsize)
        nn = CTRNN_Lauren.CTRNN(size)
        nn.randomizeParameters()
        nn.setTimeConstantVector(t)
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
    
#Research Question: How does changing the time constant for the whole network affect neural output?


plt.plot(Possible_Time_Constants, NewTotalAvgList)
#plt.errorbar(possible_connections,NewTotalAvgList,yerr=Errors, label = 'average over each possible connection', fmt='o')
plt.xlabel("Possible Time Constants (for all neurons)")
plt.ylabel("Output")
plt.title("Neural activity")
#plt.legend(loc='lower right')
plt.show()        
