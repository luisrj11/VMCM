
# Packes
from random import random, seed
from time import time
from math import exp, sqrt
#from decimal import *
import numpy as np

from VMCM.Hamiltonian.LocalEnergy import LocalEnergy
from VMCM.utils.Type import Type

'''
===============================================================================
Metropolis algorithm

To instance this class always has to be necessary pass a NUmber particles and
dimencion of the system.

Type_calculations: refers to calculation of the local energy can be numeric or 
anlytic. The nuemric case we calculate the nuemric derivate local energy and 
analytic case just use analytical expression.

for : 
    alpha
    Number_MC_cycles
    Step_size_jumping

can be instance when calls a class but it is not necesary becuase you will be
able to configure when call the method metropolis_algorithm
===============================================================================
'''

class Metropolis(LocalEnergy):

    def __init__(self, Number_particles, Dimension,Type_calculations,alpha = None,Number_MC_cycles = None,Step_size_jumping = None) -> None:
        super().__init__(Number_particles, Dimension)
        self.Type_calculations = Type_calculations 
        self.alpha = alpha
        self.Number_MC_cycles = Number_MC_cycles
        self.Step_size_jumping = Step_size_jumping
    '''
    ===============================================================================
    Metropolis algorithm 
    ===============================================================================
    '''
    
    def metropolis_algorithm(self, 
                             alpha :Type.Vector | Type.Float=None,          # Parameter          
                             Number_MC_cycles : Type.Int = None,            # NUmber of Monte Carlos cycles
                             Step_size_jumping : Type.Float = 1.0,          # Step size jumping of the randon number(walker)
                                )-> tuple:
        
        '''
        Setting the number particles and dimension and verify if you instance directly
        the configuration of the parameter for metropolis_algorithm method
        '''
        if  self.Number_MC_cycles != None and Number_MC_cycles == None :
                Number_MC_cycles = self.Number_MC_cycles

        if self.Step_size_jumping != None and Step_size_jumping == None :
                Step_size_jumping = self.Step_size_jumping

        if self.alpha != None and alpha == None:
                alpha = self.alpha

        if Number_MC_cycles == None or Step_size_jumping == None or alpha == None:
                print('<rror> you have to set up Number_MC_cycles or step paramter or alpha')
                return TypeError
        
        Number_particles = self.Number_particles         # Number particles    
        Dimension = self.Dimension                       # Dimension

        # Starting time consuming
        Time_inicio = time()

        # What function use analytic or numeric for wave function and local energy
        if type(self.Type_calculations) == str:
            if self.Type_calculations == 'analytic':

                # Wave function objectType_calculations
                wave_function = super().wave_function                  

                # Local energy object
                local_energy = super().local_energy_analytics 
            
            elif self.Type_calculations == 'numeric':

                # Wave function object
                wave_function = super().ln_wave_function                  

                # Local energy object
                local_energy =super().local_energy_num 

            
            elif self.Type_calculations == 'interacting' :
                
                # Wave function objectType_calculations
                wave_function = super().ln_interaction_wave_function   
                
                if self.Type_calculations == 'analytic':               

                    # Local energy object
                    local_energy = super().local_energy_interaction_analytic  

                elif self.Type_calculations == 'numeric':   

                    # Local energy object
                    local_energy = super().local_energy_interaction_numeric  

                print('No implement yet')

            else: 
                print('<error>: Type_calculations only takes {analytic, numeric, interacting}')
                return NameError

        else:
            print('<error>: Type_calculations parameter has to be a string type and onaly takes {analytic, numeric, interacting}')
            return TypeError

        # Sava infomations  
        energy = 0.0  
        energy2 = 0.0 

        # Positions 
        Position_old = np.zeros((Number_particles,Dimension), np.double)
        Position_new = np.zeros((Number_particles,Dimension), np.double)

        # seed starts random numbers  
        seed()

        # Initial position
        for i in range(Number_particles):
            for j in range(Dimension):
                Position_old[i][j] = Step_size_jumping * (random() - .5)
        wfold = wave_function(Position_old,alpha)
        if self.Type_calculations == 'numeric':
            wfold = exp(wfold)
        
        # Loop over Monte Carlos cicles (MCcycles)
        for MCcycle in range(Number_MC_cycles):
            
            # Trial position moving one particle at the time
            for i in range(Number_particles):
                for j in range(Dimension):
                    Position_new[i][j] = Position_old[i][j] + Step_size_jumping * (random() - .5)
                wfnew = wave_function(Position_new,alpha)
                if self.Type_calculations == 'numeric':
                    wfnew = exp(wfnew)
                
                # Metropolis test to see whether we accept 
                if random() < wfnew**2 / wfold**2:
                    for j in range(Dimension):
                        Position_old[i][j] = Position_new[i][j]
                    wfold = wfnew
            
            # Calculate the local energy 
            DeltaE = local_energy(Position_old,alpha)
            energy += DeltaE
            energy2 += DeltaE**2
    
        # Calculate mean, variance and error  
        energy /= Number_MC_cycles
        energy2 /= Number_MC_cycles
        variance = energy2 - energy**2
        error = sqrt(variance/Number_MC_cycles)

        # Time CPU
        Time_fin = time()
        Time_consuming = Time_fin - Time_inicio
        
        return energy, variance, error, Time_consuming
    

if __name__ == "__main__":
    print('PROGRAM RUNNING IN THE CURRENT FILE')