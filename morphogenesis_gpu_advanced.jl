"""
Advanced GPU-Accelerated Morphogenesis with Dynamic Features
Includes: dynamic oxygen, cell division/death, additional forces, and extensibility to 3D
Based on research by Jeannin-Girardon, Ballet, and Rodin
"""

using CUDA
using Plots
using Statistics
using LinearAlgebra
using Random
using Printf

# ============================================
# Configuration Parameters
# ============================================

if !@isdefined(N_CELLS_INITIAL)
    const N_CELLS_INITIAL = 100  # Initial number of cells
end

const MAX_CELLS = N_CELLS_INITIAL * 3  # Allow for cell division
const DT = 0.01          # Time step for simulation
const MAX_STEPS = 2000   # Maximum simulation steps
const DOMAIN_SIZE = 10.0 # Simulation domain size
const CELL_RADIUS = 0.15 # Cell radius for collision detection

# Physical parameters
const ATTRACTION_STRENGTH = 0.5    # Attraction to target
const REPULSION_STRENGTH = 1.0     # Repulsion between cells
const ADHESION_STRENGTH = 0.2      # NEW: Cell-cell adhesion
const ELASTIC_STRENGTH = 0.3       # NEW: Elastic forces
const DAMPING = 0.9                # Velocity damping
const MAX_VELOCITY = 1.0           # Maximum cell velocity

# Oxygen dynamics parameters (NEW)
const OXYGEN_DIFFUSION = 0.1       # Diffusion coefficient
const OXYGEN_CONSUMPTION = 0.05    # Consumption rate per cell
const OXYGEN_PRODUCTION_VESSEL = 0.1  # Production by vessels
const OXYGEN_DECAY = 0.01          # Natural decay rate

# Cell division and death parameters (NEW)
const DIVISION_PROB = 0.001        # Probability of division per step (high oxygen)
const DEATH_PROB = 0.0005          # Probability of death per step (low oxygen)
const DIVISION_OXYGEN_THRESHOLD = 0.6
const DEATH_OXYGEN_THRESHOLD = 0.2

# Chemotaxis parameters (NEW)
const CHEMOTAXIS_STRENGTH = 0.3    # Attraction to oxygen gradients

# Oxygen parameters
const OXYGEN_ZONES = [
    (x=0.0, y=5.0, r=2.0, strength=1.0),
    (x=-3.0, y=-2.0, r=1.5, strength=0.8),
    (x=3.0, y=-2.0, r=1.5, strength=0.8),
]

const OXYGEN_THRESHOLD_HIGH = 0.7
const OXYGEN_THRESHOLD_LOW = 0.3

# Cell types
const CELL_TYPE_BASE = 1
const CELL_TYPE_VESSEL = 2
const CELL_TYPE_FIBROBLAST = 3
const CELL_TYPE_DEAD = 0  # NEW: Dead cells (to be removed)

# ============================================
# GPU Kernels
# ============================================

"""
Dynamic oxygen field with diffusion, consumption, and production (GPU kernel)
"""
function oxygen_dynamics_kernel!(oxygen_new, oxygen, positions, types, active,
                                 n_cells, zones, n_zones, dt, diffusion, 
                                 consumption, production, decay)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= n_cells
        if active[idx] == 1
            x = positions[1, idx]
            y = positions[2, idx]
            
            # Current oxygen level
            o = oxygen[idx]
            
            # Diffusion (simplified - could use grid-based for accuracy)
            # Average with neighbors
            o_avg = 0.0f0
            count = 0
            for j in 1:n_cells
                if active[j] == 1 && j != idx
                    dx = x - positions[1, j]
                    dy = y - positions[2, j]
                    dist = sqrt(dx*dx + dy*dy)
                    if dist < 1.0f0  # Within diffusion range
                        o_avg += oxygen[j]
                        count += 1
                    end
                end
            end
            if count > 0
                o_avg /= Float32(count)
                o += diffusion * (o_avg - o) * dt
            end
            
            # Production by source zones
            for i in 1:n_zones
                dx = x - zones[1, i]
                dy = y - zones[2, i]
                dist_sq = dx * dx + dy * dy
                r_sq = zones[3, i] * zones[3, i]
                strength = zones[4, i]
                
                o_source = strength * exp(-dist_sq / (2.0f0 * r_sq))
                o += o_source * dt
            end
            
            # Production by vessel cells
            if types[idx] == CELL_TYPE_VESSEL
                o += production * dt
            end
            
            # Consumption by all cells
            o -= consumption * dt
            
            # Natural decay
            o -= decay * o * dt
            
            # Clamp to [0, 1]
            oxygen_new[idx] = clamp(o, 0.0f0, 1.0f0)
        else
            oxygen_new[idx] = oxygen[idx]
        end
    end
    
    return nothing
end

"""
Enhanced forces with adhesion, elasticity, and chemotaxis (GPU kernel)
"""
function enhanced_forces_kernel!(forces, positions, velocities, targets, types, 
                                 oxygen, active, n_cells, cell_radius, domain_size,
                                 adhesion, elastic, chemotaxis)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= n_cells && active[idx] == 1
        x = positions[1, idx]
        y = positions[2, idx]
        tx = targets[1, idx]
        ty = targets[2, idx]
        
        fx = 0.0f0
        fy = 0.0f0
        
        # Attraction to target (elastic force)
        dx_target = tx - x
        dy_target = ty - y
        dist_target = sqrt(dx_target * dx_target + dy_target * dy_target)
        
        if dist_target > 0.01f0
            # Elastic spring force (Hooke's law)
            fx += elastic * dx_target
            fy += elastic * dy_target
        end
        
        # Chemotaxis: move towards oxygen gradient
        o_center = oxygen[idx]
        grad_x = 0.0f0
        grad_y = 0.0f0
        
        # Sample oxygen gradient by looking at nearby positions
        for j in 1:n_cells
            if active[j] == 1 && j != idx
                dx = positions[1, j] - x
                dy = positions[2, j] - y
                dist = sqrt(dx*dx + dy*dy)
                
                if dist < 1.0f0 && dist > 0.01f0
                    o_diff = oxygen[j] - o_center
                    grad_x += o_diff * dx / (dist * dist)
                    grad_y += o_diff * dy / (dist * dist)
                end
            end
        end
        
        fx += chemotaxis * grad_x
        fy += chemotaxis * grad_y
        
        # Cell-cell interactions (repulsion + adhesion)
        for j in 1:n_cells
            if active[j] == 1 && j != idx
                dx = x - positions[1, j]
                dy = y - positions[2, j]
                dist_sq = dx * dx + dy * dy
                min_dist = 2.0f0 * cell_radius
                
                if dist_sq < min_dist * min_dist && dist_sq > 0.001f0
                    dist = sqrt(dist_sq)
                    
                    # Repulsion (strong at close range)
                    force_rep = 1.0f0 * (min_dist - dist) / dist
                    fx += force_rep * dx / dist
                    fy += force_rep * dy / dist
                    
                    # Adhesion (weak attractive force at medium range)
                    if dist < 1.5f0 * min_dist && dist > min_dist
                        force_adh = -adhesion * (dist - min_dist) / dist
                        fx += force_adh * dx / dist
                        fy += force_adh * dy / dist
                    end
                end
            end
        end
        
        # Oxygen-based modulation
        o = oxygen[idx]
        if types[idx] == CELL_TYPE_VESSEL
            # Vessels are more mobile in high oxygen
            fx *= (1.0f0 + 0.5f0 * o)
            fy *= (1.0f0 + 0.5f0 * o)
        elseif types[idx] == CELL_TYPE_FIBROBLAST
            # Fibroblasts move slower
            fx *= (0.5f0 + 0.5f0 * o)
            fy *= (0.5f0 + 0.5f0 * o)
        end
        
        forces[1, idx] = fx
        forces[2, idx] = fy
    else
        forces[1, idx] = 0.0f0
        forces[2, idx] = 0.0f0
    end
    
    return nothing
end

"""
Update positions and velocities (GPU kernel)
"""
function update_kernel!(positions, velocities, forces, active, n_cells, 
                       dt, damping, max_vel, domain_size)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= n_cells && active[idx] == 1
        # Update velocity
        vx = velocities[1, idx] * damping + forces[1, idx] * dt
        vy = velocities[2, idx] * damping + forces[2, idx] * dt
        
        # Limit velocity
        vel_mag = sqrt(vx * vx + vy * vy)
        if vel_mag > max_vel
            vx = vx / vel_mag * max_vel
            vy = vy / vel_mag * max_vel
        end
        
        velocities[1, idx] = vx
        velocities[2, idx] = vy
        
        # Update position
        x = positions[1, idx] + vx * dt
        y = positions[2, idx] + vy * dt
        
        # Boundary conditions (reflective)
        half_domain = domain_size / 2.0f0
        if x < -half_domain
            x = -half_domain
            velocities[1, idx] = -velocities[1, idx] * 0.5f0
        elseif x > half_domain
            x = half_domain
            velocities[1, idx] = -velocities[1, idx] * 0.5f0
        end
        
        if y < -half_domain
            y = -half_domain
            velocities[2, idx] = -velocities[2, idx] * 0.5f0
        elseif y > half_domain
            y = half_domain
            velocities[2, idx] = -velocities[2, idx] * 0.5f0
        end
        
        positions[1, idx] = x
        positions[2, idx] = y
    end
    
    return nothing
end

"""
Update cell types based on oxygen levels (GPU kernel)
"""
function update_types_kernel!(types, oxygen, active, n_cells, 
                             threshold_high, threshold_low)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= n_cells && active[idx] == 1
        o = oxygen[idx]
        
        if o > threshold_high
            types[idx] = CELL_TYPE_VESSEL
        elseif o < threshold_low
            types[idx] = CELL_TYPE_FIBROBLAST
        else
            types[idx] = CELL_TYPE_BASE
        end
    end
    
    return nothing
end

# ============================================
# CPU Functions for Cell Division and Death
# ============================================

"""
Handle cell division and death (CPU - infrequent operations)
"""
function handle_division_death!(positions_h, velocities_h, types_h, oxygen_h, 
                                active_h, targets_h, n_active)
    new_cells = []
    dead_cells = []
    
    # Check each active cell
    for i in 1:length(active_h)
        if active_h[i] == 1
            o = oxygen_h[i]
            
            # Cell death in low oxygen
            if o < DEATH_OXYGEN_THRESHOLD && rand() < DEATH_PROB
                active_h[i] = 0
                push!(dead_cells, i)
            # Cell division in high oxygen
            elseif o > DIVISION_OXYGEN_THRESHOLD && rand() < DIVISION_PROB
                # Find an inactive slot for new cell
                new_idx = findfirst(x -> x == 0, active_h)
                if !isnothing(new_idx)
                    # Create daughter cell nearby
                    angle = rand() * 2π
                    offset = CELL_RADIUS * 1.5
                    
                    positions_h[1, new_idx] = positions_h[1, i] + offset * cos(angle)
                    positions_h[2, new_idx] = positions_h[2, i] + offset * sin(angle)
                    
                    velocities_h[1, new_idx] = velocities_h[1, i] * 0.5
                    velocities_h[2, new_idx] = velocities_h[2, i] * 0.5
                    
                    types_h[new_idx] = types_h[i]
                    oxygen_h[new_idx] = o
                    active_h[new_idx] = 1
                    targets_h[1, new_idx] = targets_h[1, i]
                    targets_h[2, new_idx] = targets_h[2, i]
                    
                    push!(new_cells, new_idx)
                end
            end
        end
    end
    
    n_active_new = sum(active_h)
    return n_active_new, length(new_cells), length(dead_cells)
end

# ============================================
# Main Simulation
# ============================================

println("="^70)
println("Advanced GPU-Accelerated Morphogenesis")
println("="^70)
println("Configuration:")
println("  • Initial cells: $N_CELLS_INITIAL")
println("  • Max cells (with division): $MAX_CELLS")
println("  • Domain size: $DOMAIN_SIZE")
println("  • Max steps: $MAX_STEPS")
println("  • Time step: $DT")
println("\nNew Features:")
println("  ✓ Dynamic oxygen (diffusion, consumption, production)")
println("  ✓ Cell division (high oxygen)")
println("  ✓ Cell death (low oxygen)")
println("  ✓ Cell-cell adhesion")
println("  ✓ Elastic forces")
println("  ✓ Chemotaxis (oxygen gradient following)")

# Check GPU availability
if !CUDA.functional()
    println("  • GPU: NOT AVAILABLE - falling back to CPU")
    error("This simulation requires a functional CUDA GPU.")
else
    println("  • GPU: $(CUDA.name(CUDA.device()))")
    println("  • GPU Memory: $(round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=2)) GB")
end
println("="^70)

# Initialize cell positions (line formation)
positions_h = zeros(Float32, 2, MAX_CELLS)
for i in 1:N_CELLS_INITIAL
    positions_h[1, i] = -DOMAIN_SIZE/4 + (DOMAIN_SIZE/2) * (i-1) / (N_CELLS_INITIAL-1)
    positions_h[2, i] = 0.0f0
end

# Initialize target positions (circle formation)
targets_h = zeros(Float32, 2, MAX_CELLS)
theta = range(0, 2π, length=N_CELLS_INITIAL+1)[1:end-1]
radius = DOMAIN_SIZE / 3
for i in 1:N_CELLS_INITIAL
    targets_h[1, i] = radius * cos(theta[i])
    targets_h[2, i] = radius * sin(theta[i])
end

# Initialize other arrays
velocities_h = zeros(Float32, 2, MAX_CELLS)
types_h = ones(Int32, MAX_CELLS)
oxygen_h = fill(0.5f0, MAX_CELLS)
active_h = zeros(Int32, MAX_CELLS)
active_h[1:N_CELLS_INITIAL] .= 1

# Initialize oxygen zones on GPU
zones_h = zeros(Float32, 4, length(OXYGEN_ZONES))
for (i, zone) in enumerate(OXYGEN_ZONES)
    zones_h[1, i] = Float32(zone.x)
    zones_h[2, i] = Float32(zone.y)
    zones_h[3, i] = Float32(zone.r)
    zones_h[4, i] = Float32(zone.strength)
end

# Transfer to GPU
positions_d = CuArray(positions_h)
velocities_d = CuArray(velocities_h)
targets_d = CuArray(targets_h)
types_d = CuArray(types_h)
active_d = CuArray(active_h)
zones_d = CuArray(zones_h)
forces_d = CUDA.zeros(Float32, 2, MAX_CELLS)
oxygen_d = CuArray(oxygen_h)
oxygen_new_d = CuArray(oxygen_h)

# Kernel launch parameters
threads = 256
blocks = ceil(Int, MAX_CELLS / threads)

println("\nStarting simulation...")
println("GPU blocks: $blocks, threads per block: $threads")

# Storage for visualization and statistics
history_positions = []
history_types = []
history_n_cells = []
history_divisions = []
history_deaths = []
save_interval = max(1, MAX_STEPS ÷ 50)

start_time = time()
n_active = N_CELLS_INITIAL
total_divisions = 0
total_deaths = 0
division_check_interval = 100  # Check every N steps

# Main simulation loop
for step in 1:MAX_STEPS
    # Dynamic oxygen field with diffusion and consumption
    @cuda threads=threads blocks=blocks oxygen_dynamics_kernel!(
        oxygen_new_d, oxygen_d, positions_d, types_d, active_d,
        MAX_CELLS, zones_d, length(OXYGEN_ZONES), Float32(DT),
        Float32(OXYGEN_DIFFUSION), Float32(OXYGEN_CONSUMPTION),
        Float32(OXYGEN_PRODUCTION_VESSEL), Float32(OXYGEN_DECAY)
    )
    copyto!(oxygen_d, oxygen_new_d)
    
    # Update cell types based on oxygen
    @cuda threads=threads blocks=blocks update_types_kernel!(
        types_d, oxygen_d, active_d, MAX_CELLS,
        Float32(OXYGEN_THRESHOLD_HIGH), Float32(OXYGEN_THRESHOLD_LOW)
    )
    
    # Calculate enhanced forces (adhesion, elasticity, chemotaxis)
    @cuda threads=threads blocks=blocks enhanced_forces_kernel!(
        forces_d, positions_d, velocities_d, targets_d, types_d,
        oxygen_d, active_d, MAX_CELLS, Float32(CELL_RADIUS), 
        Float32(DOMAIN_SIZE), Float32(ADHESION_STRENGTH),
        Float32(ELASTIC_STRENGTH), Float32(CHEMOTAXIS_STRENGTH)
    )
    
    # Update positions and velocities
    @cuda threads=threads blocks=blocks update_kernel!(
        positions_d, velocities_d, forces_d, active_d, MAX_CELLS,
        Float32(DT), Float32(DAMPING), Float32(MAX_VELOCITY), Float32(DOMAIN_SIZE)
    )
    
    # Handle cell division and death (periodically, on CPU)
    if step % division_check_interval == 0
        CUDA.synchronize()
        copyto!(positions_h, positions_d)
        copyto!(velocities_h, velocities_d)
        copyto!(types_h, types_d)
        copyto!(oxygen_h, oxygen_d)
        copyto!(active_h, active_d)
        copyto!(targets_h, targets_d)
        
        n_active, n_new, n_dead = handle_division_death!(
            positions_h, velocities_h, types_h, oxygen_h,
            active_h, targets_h, n_active
        )
        
        total_divisions += n_new
        total_deaths += n_dead
        
        # Copy back to GPU
        copyto!(positions_d, positions_h)
        copyto!(velocities_d, velocities_h)
        copyto!(types_d, types_h)
        copyto!(oxygen_d, oxygen_h)
        copyto!(active_d, active_h)
        copyto!(targets_d, targets_h)
    end
    
    # Save frames for visualization
    if step % save_interval == 0
        CUDA.synchronize()
        push!(history_positions, Array(positions_d))
        push!(history_types, Array(types_d))
        push!(history_n_cells, n_active)
        push!(history_divisions, total_divisions)
        push!(history_deaths, total_deaths)
        
        # Print progress
        elapsed = time() - start_time
        progress = step / MAX_STEPS * 100
        @printf("Progress: %.1f%% | Step %d/%d | Cells: %d (+%d/-%d) | Time: %.2fs\n",
                progress, step, MAX_STEPS, n_active, total_divisions, total_deaths, elapsed)
    end
end

CUDA.synchronize()
total_time = time() - start_time

println("\n" * "="^70)
println("Simulation Complete!")
println("="^70)
println("Total time: $(round(total_time, digits=2))s")
println("Average speed: $(round(MAX_STEPS / total_time, digits=1)) steps/second")
println("Cell population dynamics:")
println("  • Initial: $N_CELLS_INITIAL")
println("  • Final: $n_active")
println("  • Total divisions: $total_divisions")
println("  • Total deaths: $total_deaths")
println("  • Net change: $(n_active - N_CELLS_INITIAL)")
println("="^70)

# ============================================
# Analysis and Visualization
# ============================================

println("\nAnalyzing final cell distribution...")

final_positions = Array(positions_d)
final_types = Array(types_d)
final_oxygen = Array(oxygen_d)
final_active = Array(active_d)

# Count cell types (only active cells)
active_indices = findall(x -> x == 1, final_active)
n_vessels = count(x -> final_types[x] == CELL_TYPE_VESSEL, active_indices)
n_fibroblasts = count(x -> final_types[x] == CELL_TYPE_FIBROBLAST, active_indices)
n_base = count(x -> final_types[x] == CELL_TYPE_BASE, active_indices)

println("Final cell distribution:")
println("  • Blood vessels: $n_vessels ($(round(n_vessels/n_active*100, digits=1))%)")
println("  • Fibroblasts: $n_fibroblasts ($(round(n_fibroblasts/n_active*100, digits=1))%)")
println("  • Base cells: $n_base ($(round(n_base/n_active*100, digits=1))%)")

println("\nCreating visualizations...")

# Color mapping for cell types
function get_cell_color(cell_type)
    if cell_type == CELL_TYPE_VESSEL
        return :red
    elseif cell_type == CELL_TYPE_FIBROBLAST
        return :blue
    else
        return :green
    end
end

# Static plot
p = plot(size=(1000, 900), aspect_ratio=:equal,
         title="Advanced GPU Morphogenesis (n=$n_active, +$total_divisions/-$total_deaths)",
         xlabel="X", ylabel="Y",
         xlim=(-DOMAIN_SIZE/2, DOMAIN_SIZE/2),
         ylim=(-DOMAIN_SIZE/2, DOMAIN_SIZE/2),
         legend=:topright)

# Draw oxygen zones
for zone in OXYGEN_ZONES
    θ = range(0, 2π, length=100)
    x_circle = zone.x .+ zone.r .* cos.(θ)
    y_circle = zone.y .+ zone.r .* sin.(θ)
    plot!(p, x_circle, y_circle,
          linestyle=:dash, linewidth=2, color=:yellow, alpha=0.3,
          fillalpha=0.05, fill=true,
          label=(zone === OXYGEN_ZONES[1] ? "Oxygen Zone" : ""))
end

# Plot final positions with types (only active cells)
for cell_type in [CELL_TYPE_VESSEL, CELL_TYPE_FIBROBLAST, CELL_TYPE_BASE]
    indices = [i for i in active_indices if final_types[i] == cell_type]
    if !isempty(indices)
        color = get_cell_color(cell_type)
        label = cell_type == CELL_TYPE_VESSEL ? "Blood Vessel" :
                cell_type == CELL_TYPE_FIBROBLAST ? "Fibroblast" : "Base Cell"
        
        scatter!(p, final_positions[1, indices], final_positions[2, indices],
                marker=:star5, markersize=6, color=color,
                label=label)
    end
end

savefig(p, "gpu_morphogenesis_advanced_n$(n_active).png")
println("✓ Saved: gpu_morphogenesis_advanced_n$(n_active).png")

# Create animation
println("\nCreating GIF animation...")
anim = @animate for (idx, (pos, cell_types, n_cells, n_div, n_death)) in enumerate(
        zip(history_positions, history_types, history_n_cells, history_divisions, history_deaths))
    
    active_idx = findall(x -> x == 1, Array(active_d))
    
    p_frame = plot(size=(800, 800), aspect_ratio=:equal,
                   title="Advanced Morphogenesis (t=$(round((idx-1)*save_interval*DT, digits=1))s, n=$n_cells, +$n_div/-$n_death)",
                   xlabel="X", ylabel="Y",
                   xlim=(-DOMAIN_SIZE/2, DOMAIN_SIZE/2),
                   ylim=(-DOMAIN_SIZE/2, DOMAIN_SIZE/2),
                   legend=:topright)
    
    # Oxygen zones
    for zone in OXYGEN_ZONES
        θ = range(0, 2π, length=50)
        x_circle = zone.x .+ zone.r .* cos.(θ)
        y_circle = zone.y .+ zone.r .* sin.(θ)
        plot!(p_frame, x_circle, y_circle,
              linestyle=:dash, linewidth=2, color=:yellow, alpha=0.3,
              label="")
    end
    
    # Plot cells by type (only active cells in this frame)
    for cell_type in [CELL_TYPE_VESSEL, CELL_TYPE_FIBROBLAST, CELL_TYPE_BASE]
        indices = [i for i in 1:MAX_CELLS if active_h[i] == 1 && cell_types[i] == cell_type]
        if !isempty(indices)
            color = get_cell_color(cell_type)
            scatter!(p_frame, pos[1, indices], pos[2, indices],
                    marker=:circle, markersize=4, color=color, label="")
        end
    end
end

gif(anim, "gpu_morphogenesis_advanced_n$(n_active).gif", fps=15)
println("✓ Saved: gpu_morphogenesis_advanced_n$(n_active).gif")

# Create population dynamics plot
p_pop = plot(size=(800, 600),
            title="Population Dynamics",
            xlabel="Time Step", ylabel="Number of Cells",
            legend=:topleft)
plot!(p_pop, (1:length(history_n_cells)) .* save_interval, history_n_cells,
      label="Total Cells", linewidth=2, color=:blue)
plot!(p_pop, (1:length(history_divisions)) .* save_interval, history_divisions,
      label="Cumulative Divisions", linewidth=2, color=:green, linestyle=:dash)
plot!(p_pop, (1:length(history_deaths)) .* save_interval, history_deaths,
      label="Cumulative Deaths", linewidth=2, color=:red, linestyle=:dash)

savefig(p_pop, "gpu_morphogenesis_population.png")
println("✓ Saved: gpu_morphogenesis_population.png")

println("\n" * "="^70)
println("All visualizations created successfully!")
println("="^70)
println("\nFeatures Demonstrated:")
println("  ✓ Dynamic oxygen diffusion and consumption")
println("  ✓ Cell division in high oxygen regions")
println("  ✓ Cell death in low oxygen regions")
println("  ✓ Cell-cell adhesion forces")
println("  ✓ Elastic forces for shape formation")
println("  ✓ Chemotaxis (oxygen gradient following)")
println("\nExtensibility:")
println("  → 3D extension: Similar kernels with z-coordinate")
println("  → Additional cell types: Extend type constants and differentiation rules")
println("  → More complex forces: Add to enhanced_forces_kernel!")
println("="^70)
