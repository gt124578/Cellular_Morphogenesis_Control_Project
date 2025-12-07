"""
Advanced GPU Morphogenesis with 100 cells
Includes dynamic oxygen, division/death, adhesion, elasticity, chemotaxis
"""

const N_CELLS_INITIAL = 100

include(joinpath(@__DIR__, "morphogenesis_gpu_advanced.jl"))
