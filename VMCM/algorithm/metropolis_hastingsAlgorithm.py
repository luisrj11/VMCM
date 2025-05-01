# Packes
from random import random, seed, normalvariate
from time import time
from math import exp, sqrt
import jax
import jax.numpy as jnp
import numpy as np
import numpy as np
#from math import exp, sqrt 
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
#from decimal import *

from VMCM.hamiltonian.local_energy import LocalEnergy
from VMCM.utils.Type import Type

'''
===============================================================================
Metropolis Hastings algorithm 

 To instance this class always has to be necessary pass a NUmber particles and
dimencion of the system.

 Type_calculations: refers to calculation of the local energy can be numeric or 
anlytic. The nuemric case we calculate the nuemric derivate local energy and 
analytic case just use analytical expression.

 for : 
    alpha
    Number_MC_cycles
    Time_step

Can be instance when calls a class but it is not necesary becuase you will be
able to configure when call the method metropolis_hastings_algorithm
===============================================================================
'''

class MetropolisHastingsAlgorithm(LocalEnergy):

    def __init__(self, Number_particles, Dimension,Type_calculations,alpha =None, Number_MC_cycles = None,Time_step = None) -> None:
        super().__init__(Number_particles, Dimension)
        self.Type_calculations = Type_calculations
        self.alpha = alpha
        self.Number_MC_cycles = Number_MC_cycles
        self.Time_step = Time_step

    '''
    ===============================================================================
    Quantum force analytic
    ===============================================================================
    '''

    # Drift force
    def quantum_force_analytic(self,
        # Variables
        r,                # Positions
        alpha,            # Parameter
        ): 

        # Save quantum force for all particles    
        qforce = np.zeros((self.Number_particles,self.Dimension), np.double)
        for i in range(self.Number_particles):
            qforce[i,:] = -4*alpha*r[i,:]
        
        return qforce
    
    '''
    ===============================================================================
    Quantum force numeric 
    ===============================================================================
    '''

    
    def quantum_force_numeric(self,
        # Variables
        r,                # Positions
        alpha,            # Parameter
        ): 
      
        qforce = np.zeros((self.Number_particles,self.Dimension), np.double)

        # Define the gradient with jax ln wave function
        Grad_ln_wave_function = jax.grad(super().ln_wave_function, argnums=0)

        # Calculate the graddient for all particles
        for i in range(self.Number_particles):
            # Gradient
            qforce[i,:] = 2.0*Grad_ln_wave_function(r[i,:], alpha) 

        return qforce

        
    '''
    ===============================================================================
    Metropolis Hastings algorithm 
    ===============================================================================
    '''
    
    def metropolis_hastings_algorithm(self,  
                             alpha :Type.Vector | Type.Float=None,           # Parameter         
                             Number_MC_cycles : Type.Int = None,               # NUmber of Monte Carlos cycles
                             Time_step = 0.05,                          
                                )-> tuple:
        
        '''
        Setting the number particles and dimension and verify if you instance directly
        the configuration of the parameter for metropolis_algorithm method
        '''
        if self.Number_MC_cycles != None and Number_MC_cycles == None :
             Number_MC_cycles = self.Number_MC_cycles

        if self.Time_step != None and Time_step == None :
            Time_step = self.Time_step
        
        if self.alpha != None and alpha == None:
            alpha = self.alpha

        if Number_MC_cycles == None or Time_step == None or alpha == None:
            print('Error you have to configure Number_MC_cycles or step paramter')
            return TypeError


        Number_particles = self.Number_particles
        Dimension = self.Dimension
       

        # Time CPU
        Time_inicio = time()

        # What function use analytic or numeric for wave function and local energy
        if type(self.Type_calculations) == str:
            if self.Type_calculations == 'analytic':

                # Wave function object
                wave_function = super().wave_function 

                # Quntum force obejec
                quantum_force = self.quantum_force_analytic                 

                # Local energy object
                local_energy = super().local_energy_analytics 
            
            elif self.Type_calculations == 'numeric':

                # Wave function object
                wave_function = super().ln_wave_function                  

                # Local energy object
                local_energy = super().local_energy_num 
                
                # Quntum force objec
                quantum_force = self.quantum_force_numeric  

            
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
                print('self.Type_calculations onaly takes {analytic, numeric, interacting')

        else:
            print('self.Type_calculations parameter has to be a string type')

        # Parameters in the Fokker-Planck simulation of the quantum force
        Dif = 0.5

        # Sava infomations  
        energy = 0.0  
        energy2 = 0.0 

        # Positions 
        Position_old = np.zeros((Number_particles,Dimension), np.double)
        Position_new = np.zeros((Number_particles,Dimension), np.double)

        # Quantum force
        Quantum_force_old = np.zeros((Number_particles,Dimension), np.double)
        Quantum_force_new = np.zeros((Number_particles,Dimension), np.double)

        # seed starts random numbers  
        seed()

        # Initial position
        for i in range(Number_particles):
            for j in range(Dimension):
                Position_old[i,j] = normalvariate(0.0,1.0)*sqrt(Time_step)
        wfold = wave_function(Position_old,alpha)
        Quantum_force_old = quantum_force(Position_old,alpha)
        if self.Type_calculations == 'numeric':
            wfold = exp(wfold)
        
        # Loop over Monte Carlos cicles (MCcycles)
        for MCcycle in range(Number_MC_cycles):
            
            # Trial position moving one particle at the time
            for i in range(Number_particles):
                for j in range(Dimension):
                    Position_new[i,j] = Position_old[i,j] + normalvariate(0.0,1.0)*sqrt(Time_step)+\
                                       Quantum_force_old[i,j]*Time_step*Dif
                wfnew = wave_function(Position_new,alpha)
                Quantum_force_new = quantum_force(Position_new,alpha)
                if self.Type_calculations == 'numeric':
                    wfnew = exp(wfnew)
                
                # Greens function
                GreensFunction = 0.0
                for j in range(Dimension):
                        GreensFunction += 0.5*(Quantum_force_old[i,j]+Quantum_force_new[i,j])*\
	                              (Dif*Time_step*0.5*(Quantum_force_old[i,j]-Quantum_force_new[i,j])-\
                                   Position_new[i,j]+Position_old[i,j])
                
                # Caclulate the Green's function
                GreensFunction = exp(GreensFunction)
                ProbabilityRatio = GreensFunction*(wfnew**2/wfold**2)

                # Metropolis-Hastings test to see whether we accept the move
                if random() <= ProbabilityRatio:
                    for j in range(Dimension):
                        Position_old[i][j] = Position_new[i][j]
                        Quantum_force_old[i,j] = Quantum_force_new[i,j]
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