# packes
import numpy as np
from joblib import Parallel, delayed

from VMCM.Algorithm.Algorithms import Algorithms

'''
===============================================================================
Class Samplings:

Here it the class to creat the data base to apply the statistic technique

To use thsi class aalways has to pass all paramter with exception Step_parameter
becuase it is directly configure:

Step_parameter = 1.0  for the Metropolis
Step_parameter = 0.05 for the Metropolis Hastings

you can change when you call the method samplings
===============================================================================
'''

class Samplings(Algorithms):
    def __init__(self, Number_particles, Dimension, algorithm, Type_calculations, alpha, Number_MC_cycles, Step_parameter = None) -> None:
        self.Number_particles = Number_particles
        self.Dimension = Dimension
        self.algorithm = algorithm
        self.Type_calculations = Type_calculations
        self.alpha = alpha
        self.Number_MC_cycles = Number_MC_cycles
        self.Step_parameter = Step_parameter

    def samplings(self,Number_samplings,Number_core):
    
        algorithm = super().set_algorithm(
                               self.Number_particles,
                               self.Dimension,
                               self.algorithm, 
                               self.Type_calculations,
                               self.alpha,
                               self.Number_MC_cycles,
                               self.Step_parameter)
        
        result = Parallel(n_jobs=Number_core)(delayed(algorithm)() for Ns in range(Number_samplings))

        result = np.array(result)

        Energies = result[:,0]
        Variances = result[:,1]
        errors = result[:,2]
        Time_consuming = result[:,3]

        return Energies, Variances, errors, Time_consuming
 
