# VMCM

This project explores the application of Monte Carlo Methods (MCM) to systems involving electron and boson tracking within a harmonic Hamiltonian framework.

The Monte Carlo Method (MCM) is a stochastic approach based on random sampling of inputs to solve statistical problems [1]. It allows for numerical approximations of high-dimensional integrals, which becomes particularly important when dealing with complex many-body systems. One major advantage of MCM is its simplicity, as it does not require an extensive mathematical background to apply effectively.

In addition, the Variational Monte Carlo Method (VMCM) is employed through the construction of a trial wavefunction, $\Psi_T(R, \alpha)$. Here, $R$ and $\alpha$ can generally be vectors, where $\alpha$ represents the variational parameters that are optimized to approximate the ground state of the system [2, 3].

## Basic steps for VMCM in many-body systems

1. Find the trial wavefunction $\Psi_T(R,\alpha)$.
2. Define the local energy: $E_L(R, \alpha) = \frac{1}{\Psi_T(R, \alpha)} \hat{H} \Psi_T(R, \alpha)$.
3. Define the probability density function (PDF) as $|\Psi_T(R, \alpha)|^2$.
4. Calculate the expectation value of the energy $\langle \hat{H} \rangle$, rewritten in terms of $|\Psi_T(R, \alpha)|^2$ and $E_L(R, \alpha)$.
5. Optimize the variational parameters $\alpha$ by minimizing $\langle E[\alpha] \rangle$.
6. Find the standard deviation $\sigma(E) = \sqrt{ \langle \hat{H}^2 \rangle - \langle \hat{H} \rangle^2 }$ to evaluate the quality of the result.

# References

[1] Introduction to Monte Carlo Methods
[2] A. Sørensen et al., Phys. Rev. A 63, 023602 (2001)
[3] E. J. Mueller et al., Phys. Rev. A 71, 053610 (2005




