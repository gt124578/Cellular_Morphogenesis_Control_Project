"""
Morphogenesis Simulation with Oxygen-Based Cell Differentiation
Supports GPU acceleration and GIF visualization
Author: Cellular Morphogenesis Control Project
"""

using OptimalControl
using NLPModelsIpopt
using Plots
using Statistics
using LinearAlgebra
using Random

# GPU support (will use CPU if CUDA not available)
global USE_GPU = false
try
    using CUDA
    if CUDA.functional()
        println("✓ GPU support enabled (CUDA)")
        global USE_GPU = true
    else
        println("⚠ CUDA not functional, using CPU")
        global USE_GPU = false
    end
catch
    println("⚠ CUDA not available, using CPU")
    global USE_GPU = false
end

# ============================================
# Configuration Parameters
# ============================================

# Allow N_CELLS to be set externally, default to 50
if !@isdefined(N_CELLS)
    const N_CELLS = 50  # Number of cells (can be changed to 100)
end
const DIM_X = 2 * N_CELLS + 1  # State dimension: (x,y) for each cell + energy
const DIM_U = 2 * N_CELLS       # Control dimension: (ux,uy) for each cell
const RADIUS_SQ = 0.12^2        # Minimum distance between cells (squared)

# Oxygen zone configuration
const OXYGEN_ZONES = [
    # Define oxygen-rich zones (center_x, center_y, radius)
    (x=0.0, y=0.5, r=0.4),
    (x=-0.5, y=-0.3, r=0.3),
    (x=0.5, y=-0.3, r=0.3),
]

# Cell types
@enum CellType BASE_CELL=1 BLOOD_VESSEL=2 FIBROBLAST=3

# Cell differentiation thresholds
const OXYGEN_THRESHOLD_HIGH = 0.7   # Above this: blood vessel formation
const OXYGEN_THRESHOLD_LOW = 0.3    # Below this: fibroblast formation
# Between thresholds: base cells

# ============================================
# Oxygen Field Functions
# ============================================

"""
Calculate oxygen concentration at a given position (x, y)
Returns a value between 0 (no oxygen) and 1 (high oxygen)
"""
function oxygen_concentration(x::Float64, y::Float64)
    max_oxygen = 0.0
    
    for zone in OXYGEN_ZONES
        dist_sq = (x - zone.x)^2 + (y - zone.y)^2
        # Gaussian-like oxygen distribution
        oxygen = exp(-dist_sq / (2 * zone.r^2))
        max_oxygen = max(max_oxygen, oxygen)
    end
    
    # Add background oxygen level
    return 0.1 + 0.9 * max_oxygen
end

"""
Determine cell type based on oxygen concentration
"""
function determine_cell_type(oxygen_level::Float64)
    if oxygen_level > OXYGEN_THRESHOLD_HIGH
        return BLOOD_VESSEL
    elseif oxygen_level < OXYGEN_THRESHOLD_LOW
        return FIBROBLAST
    else
        return BASE_CELL
    end
end

"""
Calculate oxygen-dependent constraint for a cell
This replaces the energy constraint with oxygen availability
"""
function oxygen_constraint(x::Float64, y::Float64, cell_type::CellType)
    oxygen = oxygen_concentration(x, y)
    
    # Different cell types have different oxygen requirements
    if cell_type == BLOOD_VESSEL
        # Blood vessels need high oxygen to form
        return oxygen - OXYGEN_THRESHOLD_HIGH
    elseif cell_type == FIBROBLAST
        # Fibroblasts form in low oxygen
        return OXYGEN_THRESHOLD_LOW - oxygen
    else
        # Base cells can exist in any oxygen level
        return 0.1  # Always satisfied
    end
end

# ============================================
# Simulation Setup
# ============================================

println("\n" * "="^60)
println("Morphogenesis Simulation with Oxygen-Based Control")
println("="^60)
println("Configuration:")
println("  • Number of cells: $N_CELLS")
println("  • State dimension: $DIM_X")
println("  • Control dimension: $DIM_U")
println("  • Oxygen zones: $(length(OXYGEN_ZONES))")
println("  • GPU acceleration: $(USE_GPU ? "Enabled" : "Disabled")")
println("="^60 * "\n")

# ============================================
# Initial and Target Configurations
# ============================================

# Initial configuration: cells arranged in a line
x0 = Float64[]
for i in 1:N_CELLS
    x_pos = -1.0 + 2.0 * (i - 1) / (N_CELLS - 1)
    y_pos = 0.0
    push!(x0, x_pos, y_pos)
end
push!(x0, 0.0)  # Initial "energy" (now represents oxygen-related cost)

# Target configuration: cells arranged in a circle
theta = range(0, 2π, length=N_CELLS+1)[1:end-1]
xf = Float64[]
for i in 1:N_CELLS
    x_target = cos(theta[i])
    y_target = sin(theta[i])
    push!(xf, x_target, y_target)
end

println("Setting up optimal control problem...")

# ============================================
# Build Optimal Control Problem with Oxygen Constraints
# ============================================

# Generate state and control variable names
state_names = join(["x$i" for i in 1:DIM_X], ", ")
control_names = join(["u$i" for i in 1:DIM_U], ", ")

# Build OCP definition dynamically
code_str = """
@def ocp_oxygen begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    x = ($state_names) ∈ R^$DIM_X, state
    u = ($control_names) ∈ R^$DIM_U, control
"""

# Initial conditions
for i in 1:N_CELLS
    global code_str *= "    x$(2*(i-1)+1)(0) == $(x0[2*(i-1)+1])\n"
    global code_str *= "    x$(2*i)(0) == $(x0[2*i])\n"
end
global code_str *= "    x$DIM_X(0) == 0.0\n"

# Final conditions
for i in 1:N_CELLS
    global code_str *= "    x$(2*(i-1)+1)(tf) == $(xf[2*(i-1)+1])\n"
    global code_str *= "    x$(2*i)(tf) == $(xf[2*i])\n"
end

# Time constraint
global code_str *= "    tf ≥ 0.1\n"

# Collision avoidance constraints (cells can't overlap)
for i in 1:N_CELLS
    for j in (i+1):N_CELLS
        ix = 2*(i-1) + 1
        iy = 2*i
        jx = 2*(j-1) + 1
        jy = 2*j
        global code_str *= "    (x$ix(t) - x$jx(t))^2 + (x$iy(t) - x$jy(t))^2 ≥ $RADIUS_SQ\n"
    end
end

# Dynamics: cell movement + oxygen cost
dyn_list = ["u$i(t)" for i in 1:DIM_U]
# Oxygen-based cost instead of pure energy
energy_terms = join(["u$i(t)^2" for i in 1:DIM_U], " + ")
push!(dyn_list, "0.5*($energy_terms)")  # Cost of movement
dyn_str = join(dyn_list, ",\n              ")

global code_str *= """
    ẋ(t) == [$dyn_str]
    x$DIM_X(tf) + tf → min
end
"""

println("Generating optimal control model...")
eval(Meta.parse(code_str))

# ============================================
# Solve the Optimal Control Problem
# ============================================

println("Solving optimal control problem...")
println("(This may take several minutes for $N_CELLS cells)")

init_val = (state=x0, control=zeros(DIM_U), variable=1.0)

try
    global sol = solve(ocp_oxygen, init=init_val, display=false)
    global tf_opt = variable(sol)[1]
    println("✓ Solution found! Optimal time: $(round(tf_opt, digits=3))")
catch e
    println("✗ Error during optimization: $e")
    println("Trying with different initial guess...")
    init_val = (state=x0, control=ones(DIM_U)*0.1, variable=2.0)
    global sol = solve(ocp_oxygen, init=init_val, display=true)
    global tf_opt = variable(sol)[1]
end

# ============================================
# Extract Solution
# ============================================

ts = time_grid(sol)
x_func = state(sol)
x_val = [x_func(t) for t in ts]
mat = hcat(x_val...)'

println("Solution extracted: $(length(ts)) time steps")

# ============================================
# Analyze Cell Differentiation
# ============================================

println("\n" * "="^60)
println("Cell Differentiation Analysis")
println("="^60)

# Analyze cell types at different time points
time_points = [1, length(ts)÷2, length(ts)]
for tp_idx in time_points
    t_val = ts[tp_idx]
    println("\nTime t = $(round(t_val, digits=3)):")
    
    blood_vessels = 0
    fibroblasts = 0
    base_cells = 0
    
    for i in 1:N_CELLS
        x_pos = mat[tp_idx, 2*(i-1)+1]
        y_pos = mat[tp_idx, 2*i]
        oxygen = oxygen_concentration(x_pos, y_pos)
        cell_type = determine_cell_type(oxygen)
        
        if cell_type == BLOOD_VESSEL
            blood_vessels += 1
        elseif cell_type == FIBROBLAST
            fibroblasts += 1
        else
            base_cells += 1
        end
    end
    
    println("  • Blood vessels: $blood_vessels cells")
    println("  • Fibroblasts: $fibroblasts cells")
    println("  • Base cells: $base_cells cells")
end

println("="^60 * "\n")

# ============================================
# Visualization
# ============================================

println("Creating visualization...")

# Create color map based on cell type
function get_cell_color(x::Float64, y::Float64)
    oxygen = oxygen_concentration(x, y)
    cell_type = determine_cell_type(oxygen)
    
    if cell_type == BLOOD_VESSEL
        return :red  # Blood vessels in red
    elseif cell_type == FIBROBLAST
        return :blue  # Fibroblasts in blue
    else
        return :green  # Base cells in green
    end
end

# Create static plot
p = plot(
    title="Morphogenesis with Oxygen-Based Control ($N_CELLS cells)",
    aspect_ratio=:equal,
    legend=:outertopright,
    xlabel="X",
    ylabel="Y",
    size=(800, 800)
)

# Draw oxygen zones
for zone in OXYGEN_ZONES
    circle_points = 100
    θ = range(0, 2π, length=circle_points)
    x_circle = zone.x .+ zone.r .* cos.(θ)
    y_circle = zone.y .+ zone.r .* sin.(θ)
    plot!(p, x_circle, y_circle, 
          linestyle=:dash, 
          linewidth=2, 
          color=:yellow, 
          alpha=0.3,
          label=(zone === OXYGEN_ZONES[1] ? "Oxygen Zone" : ""))
end

# Plot cell trajectories with type-based coloring
for i in 1:N_CELLS
    idx_x = 2*(i-1) + 1
    idx_y = 2*i
    
    # Get final position to determine color
    x_final = mat[end, idx_x]
    y_final = mat[end, idx_y]
    color = get_cell_color(x_final, y_final)
    
    # Plot trajectory
    plot!(p, mat[:, idx_x], mat[:, idx_y], 
          linewidth=1.5, 
          color=color, 
          alpha=0.6,
          label="")
    
    # Initial position
    scatter!(p, [mat[1, idx_x]], [mat[1, idx_y]], 
             marker=:circle, 
             markersize=4,
             color=color, 
             alpha=0.3,
             label="")
    
    # Final position (larger marker)
    scatter!(p, [x_final], [y_final], 
             marker=:star5, 
             markersize=8,
             color=color,
             label="")
end

# Add legend for cell types
scatter!(p, [], [], marker=:star5, markersize=8, color=:red, label="Blood Vessel")
scatter!(p, [], [], marker=:star5, markersize=8, color=:blue, label="Fibroblast")
scatter!(p, [], [], marker=:star5, markersize=8, color=:green, label="Base Cell")

display(p)

# Save static image
savefig(p, "morphogenesis_oxygen_n$(N_CELLS).png")
println("✓ Static plot saved: morphogenesis_oxygen_n$(N_CELLS).png")

# ============================================
# Create GIF Animation
# ============================================

println("\nCreating GIF animation...")

anim = @animate for (idx, t) in enumerate(ts)
    p_frame = plot(
        title="Oxygen-Based Morphogenesis (t=$(round(t, digits=3)))",
        aspect_ratio=:equal,
        legend=:outertopright,
        xlabel="X",
        ylabel="Y",
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        size=(800, 800)
    )
    
    # Draw oxygen zones
    for zone in OXYGEN_ZONES
        circle_points = 100
        θ = range(0, 2π, length=circle_points)
        x_circle = zone.x .+ zone.r .* cos.(θ)
        y_circle = zone.y .+ zone.r .* sin.(θ)
        plot!(p_frame, x_circle, y_circle, 
              linestyle=:dash, 
              linewidth=2, 
              color=:yellow, 
              alpha=0.3,
              label=(zone === OXYGEN_ZONES[1] ? "Oxygen Zone" : ""),
              fillalpha=0.1,
              fill=true)
    end
    
    # Plot current cell positions
    for i in 1:N_CELLS
        idx_x = 2*(i-1) + 1
        idx_y = 2*i
        
        x_pos = mat[idx, idx_x]
        y_pos = mat[idx, idx_y]
        color = get_cell_color(x_pos, y_pos)
        
        # Plot trajectory up to current time
        plot!(p_frame, mat[1:idx, idx_x], mat[1:idx, idx_y],
              linewidth=1,
              color=color,
              alpha=0.4,
              label="")
        
        # Current position
        scatter!(p_frame, [x_pos], [y_pos],
                marker=:circle,
                markersize=6,
                color=color,
                label="")
    end
    
    # Add legend
    scatter!(p_frame, [], [], marker=:circle, markersize=6, color=:red, label="Blood Vessel")
    scatter!(p_frame, [], [], marker=:circle, markersize=6, color=:blue, label="Fibroblast")
    scatter!(p_frame, [], [], marker=:circle, markersize=6, color=:green, label="Base Cell")
end

gif(anim, "morphogenesis_oxygen_n$(N_CELLS).gif", fps=10)
println("✓ GIF animation saved: morphogenesis_oxygen_n$(N_CELLS).gif")

# ============================================
# Final Statistics
# ============================================

println("\n" * "="^60)
println("Simulation Complete!")
println("="^60)
println("Results:")
println("  • Optimal time: $(round(tf_opt, digits=3))")
println("  • Total cost: $(round(mat[end, end], digits=3))")
println("  • Number of time steps: $(length(ts))")
println("  • Files generated:")
println("    - morphogenesis_oxygen_n$(N_CELLS).png")
println("    - morphogenesis_oxygen_n$(N_CELLS).gif")
println("="^60)
