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
Class Energygradient it makes to calculate the derivative of the energy for the 
minimized step
===============================================================================
'''

class Energygradient(LocalEnergy):

    def __init__(self, Number_particles, Dimension,Type_calculations,Number_MC_cycles = None,Step_parameter = None) -> None:
        super().__init__(Number_particles, Dimension)
        self.Type_calculations = Type_calculations 
        self.Number_MC_cycles = Number_MC_cycles
        self.Step_size_jumping = Step_parameter
        self.Time_step = Step_parameter
    
    '''
    ===============================================================================
    Gradient of the wave function repect to the parameter alpha
    ===============================================================================
    '''

    # Derivate of wave function ansatz as function of variational parameters
    def derivate_wave_function_analytic(self,
        r,
        alpha = None):  

        r_sum = np.sum(r**2)
        return -1*r_sum
    
    # Derivate of wave function ansatz as function of variational parameters
    def derivate_ln_wave_function_numeric(self,
        r,
        alpha = None):  

        # Define the gradient with jax ln wave function repect a parameter alpha
        Grad_ln_wave_function = jax.grad(super().ln_wave_function, argnums=1)

        Tota_grad = 0.0

        # Total gradient over all particles
        for i in range(self.Number_particles):
            Evaluated_grad = Grad_ln_wave_function(r[i,:],alpha)
            Tota_grad += Evaluated_grad
        return Tota_grad
        
    '''
    ===============================================================================
    Metropolis algorithm 
    ===============================================================================
    '''
    
    def energy_derivative_metropolis(self, 
                             alpha :Type.Vector | Type.Float,           # Parameter          
                             Number_MC_cycles : Type.Int = None,        # Number of Monte Carlos cycles
                             Step_size_jumping : Type.Float = 1.0,      # Step size jumping of the randon number(walker)
                                )-> tuple:
        
        # Setting number Monte Carlos cycles and step parameter
        if  self.Number_MC_cycles != None and Number_MC_cycles == None :
                Number_MC_cycles = self.Number_MC_cycles

        if self.Step_size_jumping != None  and Step_size_jumping == None:
                Step_size_jumping = self.Step_size_jumping

        if Number_MC_cycles == None or Step_size_jumping == None :
                print('<error> you have to set up Number_MC_cycles or step paramter')
                return TypeError
        
        # Setting the number particles and dimension
        Number_particles = self.Number_particles
        Dimension = self.Dimension

        # What function use analytic or numeric for wave function and local energy
        if type(self.Type_calculations) == str:
            if self.Type_calculations == 'analytic':

                # Wave function objectType_calculations
                wave_function = super().wave_function                  

                # Local energy object
                local_energy = super().local_energy_analytics 

                # Derivative wave function object
                derivate_wave_function = self.derivate_wave_function_analytic
            
            elif self.Type_calculations == 'numeric':

                # Wave function object
                wave_function = super().ln_wave_function                  

                # Local energy object
                local_energy =super().local_energy_num 

                # Derivative wave function object
                derivate_wave_function = self.derivate_ln_wave_function_numeric

            
            elif self.Type_calculations == 'interacting' :
                    print('No implement yet')

            else: 
                print('<error>: Type_calculations only takes {analytic, numeric, interacting}')
                return NameError

        else:
            print('<error>: Type_calculations parameter has to be a string type and onaly takes {analytic, numeric, interacting}')
            return TypeError

        # Sava infomations  
        energy = 0.0
        DeltaPsi = 0.0
        DerivativePsiE = 0.0

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
            DerPsi = derivate_wave_function(Position_old,alpha)
            DeltaPsi += DerPsi
            energy += DeltaE
            DerivativePsiE += DerPsi*DeltaE

        # calculate energy and its derivative
        energy /= Number_MC_cycles
        DerivativePsiE /= Number_MC_cycles
        DeltaPsi /= Number_MC_cycles
        Energy_derivative  = 2*(DerivativePsiE-DeltaPsi*energy)

        return Energy_derivative    
    
    '''
    ===============================================================================
    Quantum force 
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
    
    def energy_darivative_metropolis_hastings(self,  
                             alpha :Type.Vector | Type.Float,           # Parameter         
                             Number_MC_cycles : Type.Int = None,               # NUmber of Monte Carlos cycles
                             Time_step = 0.05,                          
                                )-> tuple:
        
        # Setting number Monte Carlos cycles and step parameter
        if self.Number_MC_cycles != None and Number_MC_cycles == None:
             Number_MC_cycles = self.Number_MC_cycles

        if self.Time_step != None and Time_step == None:
            Time_step = self.Time_step

        if Number_MC_cycles == None or Time_step == None :
            print('<error> you have to configure Number_MC_cycles or step paramter')
            return TypeError
        
        # Setting the number particles and dimension
        Number_particles = self.Number_particles
        Dimension = self.Dimension

        # What function use analytic or numeric for wave function and local energy
        if type(self.Type_calculations) == str:
            if self.Type_calculations == 'analytic':

                # Wave function object
                wave_function = super().wave_function 

                # Quntum force obejec
                quantum_force = self.quantum_force_analytic                 

                # Local energy object
                local_energy = super().local_energy_analytics 

                # Derivative wave function object
                derivate_wave_function = self.derivate_wave_function_analytic

            
            elif self.Type_calculations == 'numeric':

                # Wave function object
                wave_function = super().ln_wave_function                  

                # Local energy object
                local_energy = super().local_energy_num 
                
                # Quntum force objec
                quantum_force = self.quantum_force_numeric  

                # Derivative wave function object
                derivate_wave_function = self.derivate_ln_wave_function_numeric

            elif self.Type_calculations == 'interacting' :
                    print('No implement yet')

            else: 
                print('self.Type_calculations onaly takes {analytic, numeric, interacting')

        else:
            print('self.Type_calculations parameter has to be a string type')

        # Parameters in the Fokker-Planck simulation of the quantum force
        Dif = 0.5

        # Sava infomations  
        DeltaPsi = 0.0
        energy = 0.0
        DerivativePsiE = 0.0

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
            DerPsi = derivate_wave_function(Position_old,alpha)
            DeltaPsi += DerPsi
            energy += DeltaE
            DerivativePsiE += DerPsi*DeltaE

        # calculate energy and its derivative
        energy /= Number_MC_cycles
        DerivativePsiE /= Number_MC_cycles
        DeltaPsi /= Number_MC_cycles
        Energy_derivative  = 2*(DerivativePsiE-DeltaPsi*energy)

        return Energy_derivative

                 