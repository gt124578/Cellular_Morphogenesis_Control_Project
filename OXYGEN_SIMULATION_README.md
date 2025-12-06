# Oxygen-Based Morphogenesis Simulation

## Overview

This simulation implements a realistic morphogenesis model where cell behavior and differentiation are controlled by oxygen availability rather than simple energy constraints. The model simulates how cells differentiate into different types (blood vessels, fibroblasts, base cells) based on oxygen zones in their environment.

## Key Features

1. **Oxygen-Based Control**: Cells differentiate based on local oxygen concentration
   - High oxygen (>0.7): Blood vessel formation
   - Low oxygen (<0.3): Fibroblast formation  
   - Medium oxygen: Base cells

2. **GPU Acceleration**: Automatic GPU support via CUDA when available
   - Falls back to CPU if GPU is not available
   - Essential for large simulations (n=50-100 cells)

3. **GIF Visualization**: Automatic generation of animated GIF showing:
   - Cell trajectories over time
   - Oxygen zones (yellow dashed circles)
   - Cell differentiation (color-coded by type)

4. **Optimal Control**: Uses OptimalControl.jl to find optimal cell movements

## Files

- `morphogenesis_oxygen_gpu.jl` - Main simulation engine
- `morphogenesis_n50.jl` - Run with 50 cells (recommended)
- `morphogenesis_n100.jl` - Run with 100 cells (requires GPU)
- `test_morphogenesis_quick.jl` - Quick test with 10 cells
- `Project.toml` - Julia dependencies

## Installation

```bash
# From the repository root
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This will install all required packages:
- OptimalControl.jl
- NLPModelsIpopt
- Plots
- CUDA (for GPU support)
- Standard library packages

## Usage

### Run with 50 cells (default configuration)
```bash
julia --project=. morphogenesis_n50.jl
```

### Run with 100 cells (requires more time/GPU)
```bash
julia --project=. morphogenesis_n100.jl
```

### Quick test with 10 cells
```bash
julia --project=. test_morphogenesis_quick.jl
```

## Output Files

Each run generates:
1. `morphogenesis_oxygen_n{N}.png` - Static plot showing final configuration
2. `morphogenesis_oxygen_n{N}.gif` - Animated visualization of the entire simulation

## Cell Differentiation Model

The simulation models realistic morphogenesis by:

1. **Oxygen Zones**: Predefined regions with high oxygen concentration
   - Simulates vasculature and oxygen supply
   - Affects cell behavior and differentiation

2. **Cell Types**:
   - **Blood Vessels** (red): Form in high-oxygen areas
     - Crucial for oxygen transport
     - Emerge near oxygen zones
   
   - **Fibroblasts** (blue): Form in low-oxygen areas
     - Support tissue structure
     - Appear in oxygen-poor regions
   
   - **Base Cells** (green): Maintain system stability
     - General-purpose cells
     - Exist across oxygen gradients

3. **Realistic Constraints**:
   - Cells cannot overlap (collision avoidance)
   - Movement has oxygen-dependent cost
   - System optimizes for minimal energy while reaching target configuration

## Performance

### CPU Performance (estimated)
- n=10: ~1-2 minutes
- n=50: ~10-30 minutes
- n=100: ~1-3 hours

### GPU Performance (estimated)
- n=50: ~5-15 minutes
- n=100: ~15-45 minutes

Note: Actual times depend on hardware and convergence characteristics.

## Technical Details

### Optimal Control Formulation

The problem is formulated as:

- **State**: (x₁, y₁, ..., xₙ, yₙ, cost) - positions of n cells + accumulated cost
- **Control**: (u₁, v₁, ..., uₙ, vₙ) - velocity commands for each cell
- **Objective**: Minimize total movement cost + time to reach target
- **Constraints**: 
  - Collision avoidance between cells
  - Oxygen-based movement costs
  - Initial and final position constraints

### Oxygen Field

Oxygen concentration is modeled as:
```
O(x,y) = 0.1 + 0.9 * max(exp(-d²/(2r²))) for all oxygen zones
```

Where d is the distance to zone center and r is the zone radius.

## Customization

To modify simulation parameters, edit `morphogenesis_oxygen_gpu.jl`:

```julia
# Change number of cells
const N_CELLS = 75  # Any value

# Modify oxygen zones
const OXYGEN_ZONES = [
    (x=0.0, y=0.5, r=0.4),  # center_x, center_y, radius
    # Add more zones...
]

# Adjust differentiation thresholds
const OXYGEN_THRESHOLD_HIGH = 0.7
const OXYGEN_THRESHOLD_LOW = 0.3
```

## Troubleshooting

### "CUDA not available"
- GPU acceleration is optional
- Simulation will run on CPU (slower but functional)
- For GPU support, install CUDA.jl properly

### Long computation times
- Use smaller n for testing (n=10-20)
- Enable GPU if available
- Consider increasing `tf ≥ 0.1` constraint if solver struggles

### Memory issues (n=100)
- GPU highly recommended for n≥50
- Close other applications
- Consider reducing collision constraints if needed

## References

- OptimalControl.jl: https://github.com/control-toolbox/OptimalControl.jl
- Original morphogenesis work: See repository README.md

## License

See LICENSE file in repository root.
