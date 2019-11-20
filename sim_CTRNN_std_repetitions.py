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
size = 3
duration = 50
stepsize = 0.1

possible_connections = np.arange(0,(size**2)+1,1)
possible_connections_ordered = np.flip(possible_connections)
NewRepOutputList = []
StdOutputList = []
ErrorsOutputList = []
for c in possible_connections_ordered:
    repetitions = np.arange(0,10,1)
    TotalAvg = 0  
    EachRepOutputList = []
    Std_Each_Rep = []
    for r in repetitions:
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
            TimeSum = TimeSum + SumNewOutput #sum of change in outputs
            step = step+1
            NewTimeSum = (TimeSum) / (duration * size) 
        EachRepOutputList.append(NewTimeSum) #This is calculating the change in neural output for every repetition
        Std_Each_Rep.append(np.std(EachRepOutputList)) #the standard deviation for each set of repetitions 
        Errors = Std_Each_Rep/(np.sqrt(len(EachRepOutputList))) #error bars for each set of repetitions 
    NewRepOutputList.append(EachRepOutputList) #this is creating a list of neural outputs for every repetition, for each possible connection.
    np.save('Neural Outputs for all repetitions, for each connection',NewRepOutputList)
    #print(NewRepOutputList[0])
    StdOutputList.append(Std_Each_Rep)
    ErrorsOutputList.append(Errors)
    


for i in np.arange(1,(size**2)+1,1):
    plt.subplot(size, size, i)
    plt.plot(repetitions,NewRepOutputList[i])
    plt.errorbar(repetitions,NewRepOutputList[i],yerr=ErrorsOutputList[i], label = 'neural output all repetitions', fmt='')
    plt.title('# Of connections:' + str(i))

plt.legend(loc='lower right')
plt.show()
    




