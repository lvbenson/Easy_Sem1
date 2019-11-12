# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:45:00 2019

@author: Lauren Benson

"""

import ctrnn_Lauren
import matplotlib.pyplot as plt
import numpy as np

# Global Parameters
size = 5
duration = 30
stepsize = 0.001
repetitions = 1

possible_connections = np.arange(0,size**2,1)
for c in possible_connections:
    time = np.arange(0.0,duration,stepsize)
    nn = ctrnn_Lauren.CTRNN(size)
    nn.randomizeParameters()
    nn.setweightmatrix3(c)
    nn.initializeState(np.zeros(size))
    outputs1 = np.zeros((len(time),size))
    step = 0 
    for t in time:
        nn.step(stepsize)
        outputs1[step] = nn.Output
        step += 1
        
outputs_array = np.array(outputs1)
#outputs_columns = np.column_stack((outputs_array))
header = "outputs"
np.savetxt('OutputVsConnections.csv', outputs_array, delimiter=',', header=header)

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
# 
# =============================================================================

