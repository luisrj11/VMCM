# Packes
from numpy.random import randint, randn
import numpy as np
from time import time

'''
===============================================================================
Class StatisticalTechniques to set the stadistic technique for better 
statistic analysis
===============================================================================
'''

class StatisticalTechniques:
    def __init__(self, Data = None) -> None:
        self.Data = Data
        
    # Returns mean of bootstrap samples                                                                                                                                                
    def statistic_apply(self,Data):
        return np.mean(Data)

    # Bootstrap algorithm                                                                                                                                                              
    def bootstrap(self,
        Data,               # Data to do bootstrap technique
        NumberResampling,   # NUmber the resampling to made a bootstrap technique 
        applay_statistics = None,  # Information you want to know about the data (Normally will be mean and variance value)
        ):
        if applay_statistics == None :
            applay_statistics = self.statistic_apply

        else:
            print('<error> you has to apply a statistic your data')
        
        t0 = time()                                     # Initialize time
        SaveDataStatidtic = np.zeros(NumberResampling); # Save data for each resampling
        NData = len(Data);                              # Number of data points

        # Non-parametric bootstrap                                                                                                                                                     
        for Nrs in range(NumberResampling):
            SaveDataStatidtic[Nrs] = applay_statistics(Data[randint(0,NData,NData)])  # Calculate the statistic for each resampling

        # Analysis result                                                                                                                                                                   
        print("Time consuming: %g sec" % (time()-t0)); 

        print("Original: ")
        print("Mean value  " , "  Variance", "  Error")
        print("%12g %15g %15g" % (np.mean(Data), np.std(Data), np.sqrt(np.std(Data/NData))))

        print("After Bootstrap technique: ")
        print("Mean value  " , "  Variance", "  Error")
        print("%12g %15g %15g" % (np.mean(SaveDataStatidtic),np.std(SaveDataStatidtic), np.sqrt(np.std(SaveDataStatidtic/NumberResampling))))

        return  SaveDataStatidtic