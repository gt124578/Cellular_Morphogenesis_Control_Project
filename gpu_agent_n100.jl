"""
GPU Agent-Based Morphogenesis with 100 cells
"""

const N_CELLS = 100

include(joinpath(@__DIR__, "morphogenesis_gpu_agent.jl"))
