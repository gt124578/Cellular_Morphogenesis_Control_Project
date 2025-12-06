"""
GPU Agent-Based Morphogenesis with 1000 cells
True scalability demonstration
"""

const N_CELLS = 1000

include(joinpath(@__DIR__, "morphogenesis_gpu_agent.jl"))
