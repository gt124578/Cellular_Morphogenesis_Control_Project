"""
GPU Agent-Based Morphogenesis with 500 cells
"""

const N_CELLS = 500

include(joinpath(@__DIR__, "morphogenesis_gpu_agent.jl"))
