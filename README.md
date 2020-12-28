# FluidSolver

## Compile and Install

### Preparation
- Install **Visual Studio**
- Install **CUDA**

### My Configuration
- Visual Studio 2017
- CUDA 10.1

### Compilation
- git clone https://github.com/demonhub/FluidSolver.git
- cd FluidSolver
- mkdir Build
- cd Build
- cmake .. -G “Visual Studio 15 2017 Win64” (for example)
- Open FluidSolver.sln
- Build Solution
- set Renderer as startup project
