# Packes
import numpy as np
import jax 
import jax.numpy as jnp
#from math import exp, sqrt 
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from VMCM.QuantuState.State import State 
from VMCM.utils.Type import Type


# Local energy
'''
===============================================================================
Define the Local energy, it is going to need the object WAVE FUNNTION, for this 
reason has to be a function of possition and parameter alpha

consideraction
- Positions always has to be a matrix [Number particles x Dimention]
- The alpha parameter can be jus a scalar o vector could be depend of the how
  mant parameter you going to need to do varaitional Monte Carlos method
===============================================================================
'''
class LocalEnergy(State):

        def __init__(self,Number_particles,Dimension) -> None:
               self.Number_particles = Number_particles
               self.Dimension = Dimension
        
        '''
        ===============================================================================
        Anlytic local energy, it is function of the position and parameter alpha
        
        r : Matrix[Number_particles, Dimension]

        alpha  : Can be a escalar or a vector when you have more that one variational 
        parameter
        ===============================================================================
        '''
        # Local energy  for the Harmonic oscillator for Np in 3D
        def local_energy_analytics(self,
            # Variables
            r : Type.Matrix,                            # Positions
            alpha : Type.Vector | Type.Float,           # Parameter
            ) -> Type.Float:

            r_sum = np.sum(r**2)

            return self.Number_particles*self.Dimension*alpha + (0.5 - 2*alpha*alpha)*r_sum
        
        '''
        ===============================================================================
        Numerical calculation of the local energy using jax packes, it is function of 
        position and paramer alpha

        Note: The wave function it is a ln(wavefunction)

        r : Matrix[Number_particles, Dimension]

        alpha  : Can be a escalar or a vector when you have more that one variational 
        parameter

        w0 : Can be a escalar or a vector when you have more that one frecuency
        ===============================================================================
        '''
        # Define the potencial energy to calculate the numerical local energy
        def potencial_energy(self,
                             r : Type.Matrix,                               # Positions 
                             frequency : Type.Vector | Type.Float ,         # Frecuencies 
                             ) -> Type.Float:
            self.frequency = frequency
            w0 = self.frequency
            return 0.5*w0*jnp.sum(r**2)
        
    
        # Define the numerical local energy         
        def local_energy_num(self,
                            r : Type.Matrix,                            # Positions(Has to be a Matrix ----> number particles x number dimension)
                            alpha : Type.Vector | Type.Float,           # parameter 
                            )-> Type.Float:
            
            # Define the gradient with jax
            Grad_ln_wave_function = jax.grad(super().ln_wave_function, argnums=0)

            # Define the hessian matrix with jax
            Hessian_ln_wave_function = jax.hessian(super().ln_wave_function,argnums = 0)

            Total_laplacian = 0.0   
            Tota_grad = 0.0

            # Calculate Laplan and gradient for all particles
            for i in range(self.Number_particles):
                
                # Gradient
                Evaluated_grad = Grad_ln_wave_function(r[i,:], alpha) 
                Tota_grad += jnp.sum(Evaluated_grad * Evaluated_grad)

                # Laplacian 
                Evaluated_hessian = Hessian_ln_wave_function(r[i,:], alpha) 
                Total_laplacian += jnp.trace(Evaluated_hessian)
        
            # Calculate potencial energy
            Potencial_energy = self.potencial_energy(r,frequency = 1.0)

            # Calculate Kinetic energy
            Kinectic_energy = -0.5*(Total_laplacian + Tota_grad)
        
            Loca_energy = Kinectic_energy + Potencial_energy

            return float(Loca_energy)
        

        def local_energy_interaction_analytic(self,
                                              ):
             pass
    
        def local_energy_interaction_numeric(self,
                                             ):
             pass