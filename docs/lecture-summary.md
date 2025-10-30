# PHY653: Computational Electromagnetics and Plasma Physics
## Course Lecture Summary

**Course**: PHY653 - Computational Electromagnetics and Plasma Physics
**Semester**: 2024/1
**Instructor**: Pakorn Wongwaitayakornkul, PhD.
**Institution**: Thammasat University, Rangsit Campus

---

## Table of Contents

1. [Lecture 1: Python Basics and Particle Projectile Simulation](#lecture-1-python-basics-and-particle-projectile-simulation)
2. [Lecture 2: Time Integration Methods and Lorentz Motion](#lecture-2-time-integration-methods-and-lorentz-motion)
3. [Lecture 3: Motion of a Single Plasma Particle](#lecture-3-motion-of-a-single-plasma-particle)
4. [Lecture 4: Finite Difference Method for Electromagnetic Fields](#lecture-4-finite-difference-method-for-electromagnetic-fields)
5. [Lecture 5: Many Particles Systems](#lecture-5-many-particles-systems)
6. [Key Concepts Overview](#key-concepts-overview)
7. [References](#references)

---

## Lecture 1: Python Basics and Particle Projectile Simulation

**Location**: `code_lab/lab1/lecture1.ipynb`

### Topics Covered

#### 1.1 Introduction to Scientific Computing with Python
- Python fundamentals for computational physics
- Basic arrays, loops, and control structures
- Plotting and visualization techniques
- Animation of physical systems

#### 1.2 Fundamental Physics Equations
- **Lorentz Force Equation**: Describes the force on a charged particle in electromagnetic fields
  ```
  F = q(E + v × B)
  ```
- **Maxwell's Equations**: Foundation of classical electromagnetism
- Self-consistent field equations in plasma physics

#### 1.3 Particle Motion in Electric Fields
- Projectile motion with constant electric fields
- Analogy to gravitational acceleration
- Energy considerations and dissipation
- Trajectory calculations and analysis

#### 1.4 Practical Examples
1. **Projectile Motion with Bouncing**: Simulating energy loss when particles bounce off surfaces
2. **Parabola of Safety**: Determining safe regions from projectile trajectories
3. **Rutherford Scattering**: Particle deflection in non-uniform Coulomb fields
4. **Alpha Particle Trajectories**: Simulation of charged particle scattering

### Learning Outcomes
- Understand basic particle dynamics in electromagnetic fields
- Implement simple numerical simulations in Python
- Visualize particle trajectories and physical phenomena
- Apply Lorentz force law to practical problems

---

## Lecture 2: Time Integration Methods and Lorentz Motion

**Location**: `code_lab/lab2/lecture2.ipynb`

### Topics Covered

#### 2.1 Numerical Integration Schemes
Understanding different methods for solving differential equations numerically:

**Euler's Method (First-Order)**
- Simplest explicit integration scheme
- First-order accuracy: O(Δt)
- Fast but accumulates significant error over time

**Leapfrog Method (Second-Order)**
- Implicit, symplectic integrator
- Second-order accuracy: O(Δt²)
- Conserves energy better than Euler method
- Alternating "kick-drift-kick" scheme

**Runge-Kutta 4 (RK4) Method**
- Fourth-order explicit method
- Very high accuracy: O(Δt⁴)
- More computationally expensive
- Excellent for general-purpose integration

#### 2.2 Lorentz Force in Crossed Fields
- Particle motion in constant perpendicular E and B fields
- **Cycloid Motion**: Characteristic trajectory pattern
- Parametric equations:
  ```
  x(t) = R(ωt - sin(ωt))
  y(t) = R(1 - cos(ωt))
  ```
  where:
  - ω = qB/m (cyclotron frequency)
  - R = mE/(qB²) (cycloid radius)

#### 2.3 Numerical Method Comparison
- Error analysis as a function of timestep
- Stability considerations
- Computational efficiency vs. accuracy tradeoffs
- When to use each method

### Key Equations
- **Lorentz Equation**: dv/dt = (q/m)(E + v × B)
- **Position Update**: dx/dt = v
- **Cyclotron Frequency**: ωc = qB/m

### Learning Outcomes
- Implement multiple numerical integration schemes
- Understand accuracy and stability of different methods
- Analyze and compare numerical errors
- Simulate particle motion in crossed electromagnetic fields

---

## Lecture 3: Motion of a Single Plasma Particle

**Location**: `code_lab/lab3/lecture3.ipynb`

### Topics Covered

#### 3.1 Guiding Center Approximation
The guiding center theory separates particle motion into:
- **Gyromotion**: Fast circular motion around field lines
- **Drift**: Slow perpendicular motion of the gyro-center

#### 3.2 Four Fundamental Drift Velocities

**1. E × B Drift**
```
vE = (E × B) / B²
```
- Independent of particle charge and mass
- Perpendicular to both E and B fields
- Common to all particles

**2. Grad-B Drift**
```
v∇B = -(m v⊥²)/(2qB³) × ∇B × B
```
- Due to non-uniform magnetic field strength
- Charge-dependent (opposite directions for ions/electrons)
- Important in magnetic confinement

**3. Curvature Drift**
- Occurs when magnetic field lines are curved
- Due to centrifugal force in the guiding center frame
- Critical for plasma confinement devices

**4. Polarization Drift**
- Higher-order drift from time-varying fields
- Important for wave-particle interactions
- Frequency-dependent effects

#### 3.3 Advanced Plasma Physics Topics

**Magnetic Mirrors**
- Particle confinement using converging magnetic fields
- Magnetic moment conservation: μ = mv⊥²/(2B)
- Loss cone and trapped particles

**Fermi Acceleration**
- Energy gain from time-dependent magnetic mirrors
- Moving mirror boundaries
- Stochastic heating mechanisms

**Wave-Particle Interactions**
- Particle motion in electromagnetic wave fields
- Resonance conditions
- Energy exchange mechanisms

#### 3.4 Numerical Simulations
- 3D visualization of particle trajectories
- Interactive plotting for complex geometries
- Implicit leapfrog for spatial and temporal-dependent fields
- Handling of complex magnetic configurations

#### 3.5 Practical Exercises
1. Uniform electromagnetic field motion
2. Line charge with magnetic field (drift verification)
3. Toroidal coil configuration (tokamak geometry)
4. Magnetic mirror confinement
5. Fermi acceleration mechanisms
6. Particle dynamics in wave fields

### Learning Outcomes
- Understand drift velocities in plasma physics
- Implement guiding center approximation numerically
- Analyze particle confinement mechanisms
- Simulate complex magnetic geometries

---

## Lecture 4: Finite Difference Method for Electromagnetic Fields

**Location**: `code_lab/lab4/lecture4.ipynb`

### Topics Covered

#### 4.1 Poisson and Laplace Equation Solvers

**2D Laplace Equation**
```
∇²φ = 0
```
- Describes potential in charge-free regions
- Boundary value problem
- Iterative solution methods

**2D Poisson Equation**
```
∇²φ = -ρ/ε₀
```
- Relates electric potential to charge distribution
- Fundamental to electrostatics
- Grid-based numerical solution

**Gauss-Seidel Method**
- Iterative relaxation technique
- Updates grid points sequentially
- Convergence criteria and acceleration
- Boundary condition handling

#### 4.2 Finite-Difference Time-Domain (FDTD) Method

**Maxwell's Equations in Differential Form**
```
∇ × E = -∂B/∂t
∇ × B = μ₀ε₀ ∂E/∂t + μ₀J
```

**Yee Grid**
- Staggered grid for E and B fields
- Spatial and temporal discretization
- Leapfrog time advancement
- Ensures numerical stability

#### 4.3 One-Dimensional FDTD Simulation

**Wave Propagation in 1D**
- Electromagnetic wave evolution
- Gaussian pulse source excitation
- Magnetic field update equations
- Electric field update equations
- Phase relationships between E and B

**Numerical Implementation**
```
B[i] = B[i] + (dt/dx) * (E[i] - E[i-1])
E[i] = E[i] + (c²dt/dx) * (B[i+1] - B[i])
```

#### 4.4 Two-Dimensional FDTD Simulation

**2D Wave Equation**
```
∂²u/∂t² = c² ∇²u
```

**Applications**
- Wave propagation in 2D media
- Double-slit interference patterns
- Boundary conditions and absorbing layers
- Source term implementation

**Key Phenomena Studied**
- Wave interference and diffraction
- Standing wave patterns
- Frequency and amplitude effects
- Material property variations
- Boundary reflections

#### 4.5 Advanced Topics
- Perfectly Matched Layers (PML) for absorbing boundaries
- Material interfaces
- Dispersion relations
- Numerical stability criteria (Courant condition)
- Convergence testing

### Learning Outcomes
- Solve Poisson's equation using finite differences
- Implement FDTD method for Maxwell's equations
- Simulate electromagnetic wave propagation
- Analyze wave phenomena (interference, diffraction)
- Understand numerical stability requirements

---

## Lecture 5: Many Particles Systems

**Location**: `code_lab/lab5/lecture5.ipynb`

### Topics Covered

#### 5.1 Gravitational N-Body Simulation

**N-Particle Interactions**
- Pairwise gravitational forces
- Acceleration calculation:
  ```
  aᵢ = G Σⱼ≠ᵢ [mⱼ(xⱼ - xᵢ)] / |xⱼ - xᵢ|³
  ```

**Physical Principles**
- Newton's law of gravitation
- Centre of mass frame analysis
- Energy conservation (kinetic + potential)

**Virial Theorem**
```
2⟨K⟩ = -⟨U⟩
```
- Relates average kinetic and potential energy
- Verification through numerical simulation
- Application to stellar systems

**Three-Body Problem**
- Chaotic dynamics
- Sensitivity to initial conditions
- No general analytical solution

#### 5.2 One-Dimensional Two-Stream Instability

**Plasma Instability Fundamentals**
- Counter-streaming electron beams
- Unmagnetized plasma medium
- Collective plasma behavior
- Growth rate analysis

**Numerical Implementation**

**1. Particle-in-Cell (PIC) Method**
- Represent plasma as computational particles
- Each particle represents many physical particles
- Track position and velocity of each particle

**2. Field Solver (1D Poisson Equation)**
```
∂²φ/∂x² = -ρ/ε₀
```
- Solved using sparse linear algebra (`scipy.sparse.linalg.spsolve`)
- Periodic or other boundary conditions
- Electric field: E = -∂φ/∂x

**3. Particle-to-Grid Coupling**
- **Weighting**: Bin particles to mesh points (calculate density)
- **Interpolation**: Electric field from grid to particle positions
- First-order (NGP) or higher-order interpolation schemes

**4. Time Integration**
- Leapfrog "kick-drift-kick" method
  - Kick: Update velocities by half timestep
  - Drift: Update positions by full timestep
  - Kick: Update velocities by another half timestep

**Initial Conditions**
- Two Gaussian beams moving in opposite directions
- Small perturbations to seed instability
- Uniform background ion charge

**Analysis Tools**
- Phase space plots (x-v diagrams)
- Electric field evolution
- Growth rate measurement
- Energy diagnostics

#### 5.3 Key Computational Techniques

**Particle Binning**
- Histogram particles onto grid
- Density calculation from particle positions
- Weighting schemes for smooth density

**Sparse Linear Solvers**
- Efficient solution of large linear systems
- Tridiagonal matrix for 1D Poisson
- Periodic boundary conditions implementation

**Visualization**
- Particle trajectories
- Phase space evolution
- Field distributions
- Time history of instability growth

### Learning Outcomes
- Implement N-body gravitational simulations
- Understand Particle-in-Cell (PIC) method
- Simulate plasma instabilities
- Analyze collective plasma phenomena
- Master particle-grid coupling techniques
- Verify physical theorems numerically

---

## Key Concepts Overview

### Numerical Methods
- **Euler Method**: Simple first-order integration
- **Leapfrog Method**: Second-order, symplectic integrator
- **Runge-Kutta 4**: Fourth-order explicit method
- **Gauss-Seidel**: Iterative solver for elliptic PDEs
- **FDTD**: Time-domain solution of Maxwell's equations
- **PIC Method**: Particle-in-Cell for kinetic plasma simulation

### Plasma Physics Concepts
- **Lorentz Force**: Fundamental force on charged particles
- **Guiding Center**: Separation of fast gyromotion and slow drift
- **Drift Velocities**: E×B, grad-B, curvature, polarization
- **Magnetic Mirrors**: Particle confinement using field gradients
- **Two-Stream Instability**: Collective plasma instability

### Electromagnetic Theory
- **Maxwell's Equations**: Foundation of electromagnetism
- **Poisson's Equation**: Electrostatic potential from charge distribution
- **Wave Propagation**: Solutions to wave equation
- **Interference and Diffraction**: Wave phenomena

### Computational Techniques
- **Finite Differences**: Discretization of differential equations
- **Grid-Based Methods**: Spatial discretization
- **Particle Tracking**: Lagrangian approach
- **Visualization**: Scientific plotting and animation
- **Error Analysis**: Understanding numerical accuracy

---

## Course Progression

The course follows a pedagogical progression from simple to complex:

1. **Lecture 1**: Foundation - Single particle in simple fields
2. **Lecture 2**: Numerical methods - Accurate time integration
3. **Lecture 3**: Advanced single particle - Complex field geometries
4. **Lecture 4**: Field solvers - Computing self-consistent fields
5. **Lecture 5**: Many particles - Collective phenomena and plasma physics

---

## References

1. **Bellan, Paul M.** *Fundamentals of Plasma Physics*. Cambridge University Press, 2008.
   - Comprehensive plasma physics textbook
   - Theoretical foundation for guiding center theory
   - Drift velocities and magnetic confinement

2. **Sadiku, Matthew NO.** *Numerical Techniques in Electromagnetics*. CRC Press, 2000.
   - Finite difference methods
   - FDTD implementation details
   - Boundary conditions and numerical stability

3. **Mocz, Philip.** *Create Your Own Plasma PIC Simulation (With Python)*. Medium, 2020.
   - Practical guide to PIC simulations
   - Python implementation examples
   - Particle-in-cell method tutorial

---

## Additional Resources

### Lecture Materials
All lecture materials are available as Jupyter notebooks in the `code_lab/` directory:
- `code_lab/lab1/lecture1.ipynb` - Python Basics & Projectile Simulation
- `code_lab/lab2/lecture2.ipynb` - Time Integration & Lorentz Motion
- `code_lab/lab3/lecture3.ipynb` - Single Plasma Particle Motion
- `code_lab/lab4/lecture4.ipynb` - Finite Difference Method
- `code_lab/lab5/lecture5.ipynb` - Many Particles Systems

### Banner Images
Visual representations of key concepts from each lecture:
- `banners/chap1-rutherford.png` - Rutherford scattering
- `banners/chap2-lorentz.png` - Lorentz force motion
- `banners/chap3-magnetic_mirror.png` - Magnetic mirror confinement
- `banners/chap4-interference.png` - Wave interference
- `banners/chap5-nbody.png` - N-body gravitational system

---

## Summary

This course provides a comprehensive introduction to computational methods in electromagnetics and plasma physics. Students learn to:

1. **Program** scientific simulations in Python
2. **Implement** various numerical methods for differential equations
3. **Understand** fundamental plasma physics phenomena
4. **Solve** Maxwell's equations using finite difference methods
5. **Simulate** complex many-body systems
6. **Analyze** numerical accuracy and stability
7. **Visualize** physical phenomena through computational experiments

The course combines theoretical understanding with practical computational skills, preparing students for research in plasma physics, computational electromagnetics, and related fields.

---

*Document created for PHY653: Computational Electromagnetics and Plasma Physics*
*Last updated: October 30, 2025*
