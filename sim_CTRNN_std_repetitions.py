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


NewTotalAvgList = []
possible_connections = np.arange(0,size**2,1)
possible_connections_ordered = np.flip(possible_connections)
for c in possible_connections:
    TotalAvg = 0
    NewTimeSumList = []
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
        NewTimeSumList.append(NewTimeSum)
        #Std = np.std(NewTimeSumList)
        #Errors = Std/np.sqrt(len(NewTimeSumList))
    NewTotalAvg = TotalAvg / repetitions #this is the average neural output over all reps (one value)
    NewTotalAvgList.append(NewTotalAvg) #This is an array of each average neural output for each number of connections
    #np.save('avg_N.O_Per_Connection', NewTotalAvgList)
    
    
#Research Question: How does changing the number of connections in the network affect neural output?


plt.plot(possible_connections, NewTotalAvgList)
#plt.errorbar(possible_connections, NewTotalAvgList, yerr=Errors, fmt='o')
plt.xlabel("Number of connections")
plt.ylabel("Change in Neural Output")
plt.title("Neural activity")
plt.show()        
        

