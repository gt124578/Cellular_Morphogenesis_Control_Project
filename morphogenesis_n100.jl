"""
Morphogenesis with n=100 cells and Oxygen-Based Control
This is the larger scale configuration requested in the problem statement
"""

# Configure for 100 cells before including main script
const N_CELLS = 100

# Include and run the main simulation
include(joinpath(@__DIR__, "morphogenesis_oxygen_gpu.jl"))
