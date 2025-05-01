# Packes
import numpy as np
from math import exp, sqrt 
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from VMCM.utils import Type

# Quantum state
'''
===============================================================================
Define the quantum state or just a trial object wave function as 'function' 
of the  particles position and varational parameter alpha

consideraction
- Positions always has to be a matrix [Mumber particles x Dimension]
- The alpha parameter can be jus a scalar o vector could be depend of the how
  mant parameter you going to need to do varaitional Monte Carlos method
===============================================================================
'''
class State:

    def __init__(self) -> None:
        pass

    '''
    ===============================================================================
    Define the wave function, it is a function of position and alpha parameters

    r : Matrix[Number_particles, Dimension]

    alpha  : Can be a escalar or a vector when you have more that one variational 
    parameter
    ===============================================================================
    '''
    
    # Wave trial wave function Harmonic oscilator
    def wave_function(self,
        # Variables
        r: Type.Matrix ,            # Positions
        alpha: Type.Vector,         # Parameter
        ) -> Type.Float:
        
        r_sum_xyz2 = np.sum(r**2)        # Squart sume of all particles

        return exp(-alpha*r_sum_xyz2)
    
    '''
    ===============================================================================
    Define the natural logarithm (ln) of the wave function, it is a function of 
    position and alpha parameters

    r : Matrix[Number_particles, Dimension]

    alpha  : Can be a escalar or a vector when you have more that one variational 
    parameter
    ===============================================================================
    '''

    # Define the ln of the trial wave function 
    def ln_wave_function(self,
        r : Type.Matrix,                            # Positions
        alpha : Type.Vector | Type.Float,           # Parameter
        ) -> Type.Float:
    
        r_sum_xyz2 = jnp.sum(r**2)

        return -alpha*r_sum_xyz2
    
    ''' 
    ===============================================================================
    Define ln of the interaction wave function 

    r : Matrix[Number_particles, Dimension]

    alpha : Can be a escalar or a vector when you have more that one variational 
    parameter
    
    rij : It is defined r_ij = |r_i - r_j|, where r_i and r_j are positions of the
    particules i and j respectively

    diameter_bosons_interaction : interaction range of the boson
    ===============================================================================
    '''

    def ln_correlation_wave_function(rij,diameter_bosons_interaction):
        if  rij <= diameter_bosons_interaction:
            return 0
        else:
            return  jnp.log(1 - diameter_bosons_interaction/rij)

    def ln_interaction_wave_function(self,
                                    r,
                                    alpha,
                                    diameter_bosons_interaction,
                                    )-> Type.Float: 
    
        Number_particles = len(r)           # NUmber particles 
        r_sum_xy2 = jnp.sum(r[:,:-1]**2)    # Squart sum of all positions for x and y direction 
        r_sum_z2 = jnp.sum(r[:,-1])         # Squart sum of all positions for z direction 

        r_sum_xyz_interacting = jnp.array([])   # Sum of all positions for correlation wave funtion
        j = 0
        for i in range(Number_particles):
            j = i + 1
            while i < j and j < Number_particles:
                rij = r[i]-r[j]     
                Norm_rij = jnp.linalg.norm(rij)
                Correlation_part = self.ln_correlation_wave_function(diameter_bosons_interaction,Norm_rij)
                r_sum_xyz_interacting = jnp.append(r_sum_xyz_interacting,Correlation_part)
                j = j + 1 
        r_sum_xyz_interacting = jnp.sum(r_sum_xyz_interacting)
        
        return -alpha[0]*r_sum_xy2 - alpha[1]*r_sum_z2 + r_sum_xyz_interacting
