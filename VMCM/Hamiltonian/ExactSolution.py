from VMCM.utils.Type import Type

# Exact energy 
'''
===============================================================================
Exact solutions for the harmonic oscillator (ho) no interactive case
===============================================================================
'''
class ExactSolution:
     
    def __init__(self,Number_particles,Dimension) -> None:
                self.Number_particles = Number_particles
                self.Dimension = Dimension
    '''
    ===============================================================================
    Exact energy solution for harmonic oscillator (ho) no interactive case
    ===============================================================================
    '''
    # The exact energy as funtion alpha values
    def exact_energy_ho_no_interact(self,
        # Variance
        alpha,                              # parameter      
        )-> None:

        N = self.Number_particles ; D = self.Dimension 
        ExactEnergies = (0.5 - 2*alpha*alpha)*((D*N)/(4*alpha)) + D*N*alpha 

        return ExactEnergies
    '''
    ===============================================================================
    Exact variance solution for harmonic oscillator (ho) no interactive case
    ===============================================================================
    '''
    # The exact variance as duntion alpha values
    def exact_variance_ho_no_interact(self,
        # Variables
        alpha : Type.Vector | Type.Float,       # Parameter
        ):  
    
        N = self.Number_particles ; D = self.Dimension 
        E2 = (1/N)*self.exact_energy_ho_no_interact(alpha)*\
                   self.exact_energy_ho_no_interact(alpha)
        
        Exact_variance = 0.0625*N*D*((D+2)/(alpha*alpha))*(0.5 - 2*alpha*alpha)**2 +\
                        0.5*N*(0.5 - 2*alpha*alpha)*D**2 + N*(D*alpha)**2  - E2  
    
        return Exact_variance   

if __name__ == "__main__":
    print('PROGRAM RUNNING IN THE CURRENT FILE')  