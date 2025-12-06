"""
Quick Test Script for Oxygen-Based Morphogenesis
Tests with n=10 cells for rapid validation
"""

# Set to 10 cells for quick testing before including the main script
const N_CELLS = 10

# Include the main script
include(joinpath(@__DIR__, "morphogenesis_oxygen_gpu.jl"))
