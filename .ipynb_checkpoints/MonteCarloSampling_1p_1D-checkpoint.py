#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

# Common imports
import os

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "Results/VMCHarmonic"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

outfile = open(data_path("VMCHarmonic.dat"),'w')

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

%matplotlib inline

# VMC for the one-dimensional harmonic oscillator
# Brute force Metropolis, no importance sampling and no energy minimization
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from decimal import *
# Trial wave function for the Harmonic oscillator in one dimension
def WaveFunction(r,alpha):
    return exp(-0.5*alpha*alpha*r*r)

# Local energy  for the Harmonic oscillator in one dimension
def LocalEnergy(r,alpha):
    return 0.5*r*r*(1-alpha**4) + 0.5*alpha*alpha

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

# The Monte Carlo sampling with the Metropolis algo
# The jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when the function is called.
def MonteCarloSampling(
    # Number Monte Carlos cicles
    NumberMCcycles= 100000,
    StepSize = 1.0,
    # Number varations of Alpha 
    VariationsAlfa = 20,
    AlphaStart= .4,
    StespAlpha = .05):
    #----------------------------------------------------------#    
    PositionOld = 0.0
    PositionNew = 0.0
    #----------------------------------------------------------#
    # Save all variations  
    Energies = np.zeros(VariationsAlfa)
    ExactEnergies = np.zeros(VariationsAlfa)
    AlphaValues = np.zeros(VariationsAlfa)
    #----------------------------------------------------------#
    # seed starts random numbers  
    seed()
    #----------------------------------------------------------#
    # Start variational parameter
    alpha = AlphaStart
    for ia in range(MaxVariations):
        alpha += StespAlpha
        AlphaValues[ia] = alpha
        energy = 0.0
        energy2 = 0.0
        #----------------------------------------------------------#
        # Initial position
        PositionOld = StepSize * (random() - .5)
        wfold = WaveFunction(PositionOld,alpha)
        #----------------------------------------------------------#
        # Loop over Monte Carlos cicles (MCcycles)
        for MCcycle in range(NumberMCcycles):
            #----------------------------------------------------------#
            #Trial position moving one particle at the time
            PositionNew = PositionOld + StepSize * (random() - .5)
            wfnew = WaveFunction(PositionNew,alpha)
            #----------------------------------------------------------#
            #Metropolis test to see whether we accept the move
            if random() < wfnew**2 / wfold**2:
                PositionOld = PositionNew
                wfold = wfnew
            DeltaE = LocalEnergy(PositionOld,alpha)
            energy += DeltaE
            energy2 += DeltaE**2
            #----------------------------------------------------------#
        #----------------------------------------------------------#
        # We calculate mean, variance and error ...
        energy /= NumberMCcycles
        energy2 /= NumberMCcycles
        variance = energy2 - energy**2
        error = sqrt(variance/NumberMCcycles)
        #----------------------------------------------------------#
        # Saving each iterations
        Energies[ia] = energy 
        Variances[ia] = variance 
        #------------ ---------------------------------------#
        # Writing in a external file("VMCHarmonic.dat") for each iterations
        # the path it is renamed with the alias(outfile)
        outfile.write('%f %f %f %f \n' %(alpha,energy,variance,error))
        #----------------------------------------------------------# 
    return Energies, AlphaValues, Variances
    
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

#Here starts the main program with variable declarations
MaxVariations = 20
Energies = np.zeros((MaxVariations))
ExactEnergies = np.zeros((MaxVariations))
ExactVariance = np.zeros((MaxVariations))
Variances = np.zeros((MaxVariations))
AlphaValues = np.zeros(MaxVariations)
(Energies, AlphaValues, Variances) = MonteCarloSampling()
outfile.close()
ExactEnergies = 0.25*(AlphaValues*AlphaValues+1.0/(AlphaValues*AlphaValues))
ExactVariance = 0.25*(1.0+((1.0-AlphaValues**4)**2)*3.0/(4*(AlphaValues**4)))-ExactEnergies*ExactEnergies

#simple subplot
plt.subplot(2, 1, 1)
plt.plot(AlphaValues, Energies, 'o-',AlphaValues, ExactEnergies,'r-')
plt.title('Energy and variance')
plt.ylabel('Dimensionless energy')
plt.subplot(2, 1, 2)
plt.plot(AlphaValues, Variances, '.-',AlphaValues, ExactVariance,'r-')
plt.xlabel(r'$\alpha$', fontsize=15)
plt.ylabel('Variance')
save_fig("VMCHarmonic")
plt.show()
#nice printout with Pandas
import pandas as pd
from pandas import DataFrame
data ={'Alpha':AlphaValues, 'Energy':Energies,'Exact Energy':ExactEnergies,'Variance':Variances,'Exact Variance':ExactVariance,}
frame = pd.DataFrame(data)
print(frame)

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#