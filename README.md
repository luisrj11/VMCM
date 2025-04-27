# VMCM
This project explores the implementation of the Variational Monte Carlo Method (VMCM) applied to systems involving electron and boson tracking within a harmonic Hamiltonian framework.

The Variational Monte Carlo Method (VMCM) is a stochastic approach based on random sampling to optimize trial wavefunctions and find approximations of ground state energies. It combines the simplicity of Monte Carlo integration with the flexibility of variational principles. The key idea is to construct a trial wavefunction, $\Psi_T(R, \alpha)$, where $R$ represents the system configuration and $\alpha$ denotes variational parameters. These parameters are optimized to minimize the expectation value of the energy, thereby approximating the ground state of the system [2, 3].

This method is particularly useful for studying many-body quantum systems due to its simplicity, numerical efficiency, and relatively low mathematical overhead.

## Basic steps for VMCM in many-body systems

1. Find the trial wavefunction $\Psi_T(R,\alpha)$.
2. Define the local energy: $E_L(R, \alpha) = \frac{1}{\Psi_T(R, \alpha)} \hat{H} \Psi_T(R, \alpha)$.
3. Define the probability density function (PDF) as $|\Psi_T(R, \alpha)|^2$.
4. Calculate the expectation value of the energy $\langle \hat{H} \rangle$, rewritten in terms of $|\Psi_T(R, \alpha)|^2$ and $E_L(R, \alpha)$.
5. Optimize the variational parameters $\alpha$ by minimizing $\langle E[\alpha] \rangle$.
6. Find the standard deviation $\sigma(E) = \sqrt{ \langle \hat{H}^2 \rangle - \langle \hat{H} \rangle^2 }$ to evaluate the quality of the result.

# References

[2] A. Sørensen et al., Phys. Rev. A 63, 023602 (2001) 

[3] E. J. Mueller et al., Phys. Rev. A 71, 053610 (2005 




