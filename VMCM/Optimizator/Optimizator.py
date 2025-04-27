
'''
===============================================================================
The optimization part
===============================================================================
'''

# packes
import numpy as np
from numpy.random import rand
from time import time

from random import seed
from scipy.optimize import minimize
import numpy as np

from VMCM.Optimizator.EnergyGradient import Energygradient
from VMCM.Optimizator.Energy import Energy

class Optimizator(Energygradient, Energy):

    def __init__(self, Number_particles, Dimension,algorithm, Type_calculations, Number_MC_cycles, Step_parameter = None) -> None:
            self.Number_particles = Number_particles
            self.Dimension = Dimension
            self.algorithm = algorithm
            self.Type_calculations = Type_calculations
            self.Number_MC_cycles = Number_MC_cycles
            self.Step_size_jumping = Step_parameter
            self.Time_step = Step_parameter


    def  function(self,Parameter):
        if self.algorithm == 'Metropolis' :
            energy = super().energy_metropolis
        
        elif self.algorithm == 'MetropolisHastings':
            energy= super().energy_metropolis_hastings

        return energy(Parameter) 

    def gradient_function(self,Parameter):
        if self.algorithm == 'Metropolis' :
            energy_gradient = super().energy_derivative_metropolis
        
        elif self.algorithm == 'MetropolisHastings':
            energy_gradient= super().energy_darivative_metropolis_hastings
        
        return energy_gradient(Parameter)
    
    
    def gradient_descent(self,bounds, iterations,learning_rate,function = None, gradient_funtion = None,):

        if function == None or gradient_funtion == None:
            function = self.function
            gradient_funtion = self.gradient_function
        
        # Starting time consuming
        Time_inicio = time()

        # Initial guess for alpha (ramdom starting point)
        Parameter = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])


        if len(Parameter) == 1:
            Parameter = float(Parameter)

        for i in range(iterations):

            # Compute the gradient at the current point
            grad = gradient_funtion(Parameter)

            # Update r using the Gradient Descent formula
            Parameter = Parameter - learning_rate * grad
     

            if Parameter < 0:
                print('< Warning > :You should try with a smaller learning rate')
                return TypeError
            
        Parameter_minimum = Parameter
        
        funtion_evaluated_minimum = function(Parameter_minimum)

        # Time CPU
        Time_fin = time()
        Time_consuming = Time_fin - Time_inicio

        return Parameter_minimum, funtion_evaluated_minimum,Time_consuming
    

    
    def minimized(self, bounds):
        # Initial guess for r (ramdom starting point)
        Parameter = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

        res = minimize(self.function, Parameter, method='BFGS', jac=self.gradient_function, options={'gtol': 1e-4,'disp': True})
        return res.x

    

    


    


        
