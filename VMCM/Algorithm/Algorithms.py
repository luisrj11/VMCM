# Packes
from random import random, seed
from time import time
from math import exp, sqrt
#from decimal import *
import numpy as np

#from VMCM.Hamiltonian.LocalEnergy import LocalEnergy
#from VMCM.utils.Type import Type
from VMCM.Algorithm.Metroplis import Metropolis
from VMCM.Algorithm.MetropolisHastingsAlgorithm import MetropolisHastingsAlgorithm
'''
===============================================================================
Algorithm use the class Methoplis and Metropolis Hastings and put togueder in 
a super class

To instance this class you do not need to pass any arguemnt because all parameter
can be configured n the method call set_algorithm.

set_algorithm is used to configure the tipe of algorithm but the other parameter too.

You could not pass alpha, Number_MC_cycles, Step_parameter because they can be
configure when you going to use the algorithm.

Step_parameter: can be represent the Step_jumping or Steptime It depends on which 
algorithms are used.

Type_calculations: refers to calculation of the local energy can be numeric or 
anlytic. The nuemric case we calculate the nuemric derivate local energy and 
analytic case just use analytical expression.

===============================================================================
'''

class Algorithms(Metropolis,MetropolisHastingsAlgorithm):
        
    def __init__(self, Number_particles = None ,Dimension = None , algorithm = None , Type_calculations = None,alpha = None,Number_MC_cycles = None, Step_parameter = None) -> None:
            self.Number_particles = Number_particles
            self.Dimension = Dimension
            self.algorithm = algorithm
            self.Type_calculations = Type_calculations
            self.Number_MC_cycles = Number_MC_cycles
            self.alpha = alpha
            self.Step_size_jumping = Step_parameter
            self.Time_step = Step_parameter

    # Setting algorithm and type of calculation
    def set_algorithm(self,Number_particles, Dimension, algorithm, Type_calculations,alpha = None, Number_MC_cycles = None, Step_parameter = None):
        self.Number_particles = Number_particles
        self.Dimension = Dimension
        self.algorithm = algorithm
        self.Type_calculations = Type_calculations
        self.Number_MC_cycles = Number_MC_cycles
        self.Step_size_jumping = Step_parameter
        self.Time_step = Step_parameter
        self.alpha = alpha
        if self.algorithm == 'Metropolis' :
            metropolis_algorithm = super().metropolis_algorithm
            return metropolis_algorithm
            
        elif self.algorithm == 'MetropolisHastings':
            metropolis_hastings_algorithm =super().metropolis_hastings_algorithm 
            return metropolis_hastings_algorithm

            
    
if __name__ == "__main__":
    print('PROGRAM RUNNING IN THE CURRENT FILE')