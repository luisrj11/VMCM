# packes
import numpy as np
from joblib import Parallel, delayed

from VMCM.algorithm.algorithms import Algorithms

'''
===============================================================================
class VaryingAlpha: 

Here it sets for the calcualtion energy varing the alpha parameter, for this part
is implemented the parallelized processes

For uses this class you always have to pass the number particles, dimension 
algorithm, type the calculation, number Monte Carlos cicles and Step_parameter 
is optional because this is configured  as:

Step_parameter = 1.0  for the Metropolis
Step_parameter = 0.05 for the Metropolis Hastings

the way used it call the method varying_alpha it is directly configured with

Variations_alfa = 20
Alpha_start = 0.1 
StespAlpha = 0.05 
Number_core =1

you can change when the methos is called
===============================================================================
'''

class VaryingAlpha(Algorithms):
    def __init__(self, Number_particles, Dimension, algorithm, Type_calculations, Number_MC_cycles, Step_parameter = None) -> None:
            self.Number_particles = Number_particles
            self.Dimension = Dimension
            self.algorithm = algorithm
            self.Type_calculations = Type_calculations
            self.Number_MC_cycles = Number_MC_cycles
            self.Step_parameter = Step_parameter

    def varying_alpha(self, Variations_alfa = 20, Alpha_start = 0.1, StespAlpha = 0.05, Number_core =1):

        algorithm = super().set_algorithm(self.Number_particles,
                                       self.Dimension,
                                       self.algorithm, 
                                       self.Type_calculations,
                                       Number_MC_cycles = self.Number_MC_cycles,
                                       Step_parameter = self.Step_parameter)

        # Save infomation
        Alpha_values = np.zeros(Variations_alfa)
        # Start variational parameter
        alpha = Alpha_start
        for ia in range(Variations_alfa):
            alpha += StespAlpha 
            Alpha_values[ia] = alpha
        
        result = Parallel(n_jobs=Number_core)(delayed(algorithm)(alpha) for alpha in Alpha_values)

        result = np.array(result)

        Energies = result[:,0]
        Variances = result[:,1]
        errors = result[:,2]
        Time_consuming = result[:,3]

        return Energies, Variances, errors, Time_consuming,Alpha_values

'''
===============================================================================
class VaryingNumberMCCycles: 

Here it sets for the calcualtion time sonsuming as funtion of number of Monte
Carlos cycles (MCc), for this part it is implemented the parallelized processes

For uses this class you always have to pass the number particles, dimension 
algorithm, type the calculation, alpha and Step_parameter is optional
because this is configured  as:

Step_parameter = 1.0  for the Metropolis
Step_parameter = 0.05 for the Metropolis Hastings

the way used it call the method varying_alpha it is directly configured with

Variation_number_MC_cycles = 6
Step = 1000
Number_core =1

you can change when the methos is called
===============================================================================
'''
    
class VaryingNumberMCCycles(Algorithms):
    def __init__(self, Number_particles, Dimension, algorithm, Type_calculations, alpha, Step_parameter = None) -> None:
        self.Number_particles = Number_particles
        self.Dimension = Dimension
        self.algorithm = algorithm
        self.Type_calculations = Type_calculations
        self.alpha = alpha
        self.Step_parameter = Step_parameter
    
    def varying_Number_MC_Cycles(self,Variation_number_MC_cycles = 6,Step = 1000, Number_core =1):
        algorithm = super().set_algorithm(self.Number_particles,
                                       self.Dimension,
                                       self.algorithm, 
                                       self.Type_calculations,
                                       alpha = self.alpha,
                                       Step_parameter = self.Step_parameter)

        All_number_MC_cycles = np.zeros(0,int)
        for ia in range(Variation_number_MC_cycles):
            All_number_MC_cycles = np.append(All_number_MC_cycles,(ia+1)*Step)
        
        result = Parallel(n_jobs=Number_core)(delayed(algorithm)(Number_MC_cycles = Number_MC_cycles) for Number_MC_cycles in All_number_MC_cycles)

        result = np.array(result)

        Energies = result[:,0]
        Variances = result[:,1]
        errors = result[:,2]
        Time_consuming = result[:,3]

        return Energies, Variances, errors, Time_consuming, All_number_MC_cycles

'''
===============================================================================
class VaringStepParameter: 

Here it sets for the calcualtion time sonsuming as funtion of number of Monte
Carlos cycles (MCc), for this part it is implemented the parallelized processes

For uses this class you always have to pass the number particles, dimension 
algorithm, type the calculation, alpha and MCc.

the way used it call the method varying_alpha it is directly configured with

Variation_step_parameter = 20
Step = 0.005
Number_core =1

you can change when the methos is called
===============================================================================
'''
    

class VaringStepParameter(Algorithms):

    def __init__(self, Number_particles, Dimension, algorithm, Type_calculations, alpha, Number_MC_cycles) -> None:
        self.Number_particles = Number_particles
        self.Dimension = Dimension
        self.algorithm = algorithm
        self.Type_calculations = Type_calculations
        self.alpha = alpha
        self.Number_MC_cycles = Number_MC_cycles
    
    def varying_step_parameter(self,Variation_step_parameter = 20,Step = 0.005, Number_core =1):
        algorithm = super().set_algorithm(self.Number_particles,
                                       self.Dimension,
                                       self.algorithm, 
                                       self.Type_calculations,
                                       alpha = self.alpha,
                                       Number_MC_cycles = self.Number_MC_cycles)

        All_step_parameter = np.zeros(0,float)
        for ia in range(Variation_step_parameter):
            All_step_parameter = np.append(All_step_parameter,(ia+1)*Step)
        
        if self.algorithm == 'Metropolis':
            result = Parallel(n_jobs=Number_core)(delayed(algorithm)(Step_size_jumping = Step_parameter) for Step_parameter in All_step_parameter)

        elif self.algorithm == 'MetropolisHastings':
            result = Parallel(n_jobs=Number_core)(delayed(algorithm)(Time_step = Step_parameter) for Step_parameter in All_step_parameter)

        else:
            print('<error> you have to configure te algorithm with Metropolis or MetropolisHastings')

        result = np.array(result)

        Energies = result[:,0]
        Variances = result[:,1]
        errors = result[:,2]
        Time_consuming = result[:,3]

        return Energies, Variances, errors, Time_consuming, All_step_parameter
 
    
 