# AQM
Advanced quantum mechanics final project

Uses the variatonal principle to estimate the Slater orbital of an electron in a hydrogen atom as a series of Gaussian orbitals. Performs gradient descent
on the amplitudes and standard deviations of these orbitals to obtain best approximation of the true orbital as a series of Gaussians, minimizing the energy
at every step.

Gradient descent is crude - esentially one small step in all directions is taken and the local tangent line (or, as this is a many-dimensional problem,
hyperplane) is estimated as a secant line between the current parameters and those one small distance away. Still, the algorithm achieves good results, with
less than 0.1% error in the case of 4 Gaussians from the theoretically optimal approximation.
