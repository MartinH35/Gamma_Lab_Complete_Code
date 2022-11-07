#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.optimize import curve_fit
from numpy import inf as INF

FEPE_NaI = [0.43666313789983, 0.22141603186894257, 0.08999789088043553, 0.012590557093286702, 0.008525107326705111, 0.11254601184415042]
Energy_Nai = [59.5409, 30.8535602, 356.02, 1173.228, 1332.492, 661.657]

Energy_CdTe = [13.81, 17.7, 59.54, 30.85, 80.9979]
FEPE_CdTe = [0.00023584268278633956, 0.00030245050801734836, 0.0006627832029187644, 0.0008416264694058657, 0.00019516927725234473]


Energy_Ge = (59.5409, 302.8508, 356.02, 1173.228, 1332.492, 661.657)
FEPE_Ge = [0.3209426457868176, 0.08617511268310077, 0.21784688625028598, 0.015633362020448762, 0.012756168956050088, 0.06831972925165693]


Energy_BGO = (59.5409,1173.228,1332.492,356.02)
FEPE_BGO = [0.4318693097316493, 0.2744373548733432, 0.24913549913002078, 0.38837443032876034]

def efficency_eqn(E, a, b, c):
    Eff = a + b*np.log(E) + c*(np.log(E))**2
    return (Eff)

def Eff_fit(x,y):
    fit, _ = curve_fit(efficency_eqn,x, y)
    a,b,c = fit
    #plt.scatter(x, y)
    x_line = np.arange(min(x), max(x), 1)
    # calculate the output for the range
    y_line = efficency_eqn(x_line, a, b, c)
    # create a line plot for the mapping function
    plt.plot(x_line, y_line, '-')
    plt.xlabel('Energy keV')
    plt.ylabel('Absolute Efficiency')
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Absolute Efficiency vs Energy For 4 Detectors')

Eff_fit(Energy_Nai,FEPE_NaI)
Eff_fit(Energy_CdTe,FEPE_CdTe)
Eff_fit(Energy_Ge,FEPE_Ge)
Eff_fit(Energy_BGO,FEPE_BGO)
plt.legend(['NaI:Tl','CdTe','HpGe','BGO'])            

 
 

