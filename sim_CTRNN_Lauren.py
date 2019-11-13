# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:21:02 2019

@author: Lauren Benson
"""

import CTRNN_Lauren
import matplotlib.pyplot as plt
import numpy as np

# Global Parameters
size = 10
duration = 50
stepsize = 0.01
repetitions = 100

NewTotalAvgList = []
possible_connections = np.arange(0,size**2,1)
for c in possible_connections:
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
      #  print(c, "connection")
       # print(r, "repetitions")
        for t in time:
            nn.step(stepsize)
            outputs1[step] = nn.Output
            NewOutput = np.absolute(outputs1[step] - outputs1[step-1])
            SumNewOutput = np.sum(NewOutput)
            TimeSum = TimeSum + SumNewOutput
            step = step+1
        NewTimeSum = (TimeSum) / (duration * size)
        TotalAvg = NewTimeSum + TotalAvg
      #  print(TotalAvg, "Total sum")
    NewTotalAvg = TotalAvg / repetitions
    NewTotalAvgList.append(NewTotalAvg)
 #   print(c, "connections ", NewTotalAvg, "Total average")
    
        
plt.plot(possible_connections, NewTotalAvgList)
plt.xlabel("Number of connections")
plt.ylabel("Output")
plt.title("Neural activity")
plt.show()        
        

#outputs_columns = np.column_stack((outputs_array))
# =============================================================================
# header = "outputs"
# np.savetxt('OutputVsConnections.csv', outputs_array, delimiter=',', header=header)
# 
# =============================================================================
# =============================================================================
# outputs_array = np.array(outputs1)
# sorted_outputs = np.sort(outputs_array)
# print(sorted_outputs)
# =============================================================================

# =============================================================================
# plt.plot(time,outputs1)
# plt.xlabel("time")
# plt.ylabel("Output")
# plt.title("Neural activity")
# plt.show()
# =============================================================================


# Run simulation

# =============================================================================
# for n in size:
#     step = 0
#     for t in time:
#         nn.step(stepsize)
#         outputs1[step] = nn.Output
#         step += 1