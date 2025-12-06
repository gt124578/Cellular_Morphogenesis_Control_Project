"""
Morphogenesis with n=50 cells and Oxygen-Based Control
This is the default configuration requested in the problem statement
"""

# Configure for 50 cells before including main script
const N_CELLS = 50

# Include and run the main simulation
include(joinpath(@__DIR__, "morphogenesis_oxygen_gpu.jl"))
