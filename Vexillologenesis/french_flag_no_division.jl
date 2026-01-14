"""
Minimal, gradient-free French Flag (Blue / White / Red)
=======================================================

Principes (inspiré des notes Fronville/Mercier) :
- Morphogenèse émergente : aucune source/gradient externe.
- Génome numérique simple : poids d'adhésion / répulsion + règle de polarité locale.
- Polarité locale : moyenne des voisins (biais gauche/droite) + relaxation.
- Différenciation : seuils sur polarité → Bleu / Blanc / Rouge.
- Contrôle MPC de forme (horizon 1) : poussée douce vers bandes cibles (centres fixés),
  tout en conservant répulsion/adhésion (règles locales). Pas d'oracle global.
- GPU-first via CUDA, repli CPU pour fumées rapides.

Usage :
    julia --project french_flag_minimal_polarity.jl
Sorties :
    - french_flag_minimal.png (état final)
    - Impression du génome (matrice d'adhésion / poids modules)
"""

using CUDA
using Random
using LinearAlgebra
using Statistics
using Printf
using Plots

# ------------------------------------------------------------
# Paramètres "génomiques" simples
# ------------------------------------------------------------
const CELL_TYPE_DEAD  = Int32(0)
const CELL_TYPE_BLUE  = Int32(1)
const CELL_TYPE_WHITE = Int32(2)
const CELL_TYPE_RED   = Int32(3)

const GENOME = (
    collision_weights = (0.0f0, 1.0f0, 1.0f0, 1.0f0),
    adhesion_weights  = (0.0f0, 1.1f0, 1.25f0, 1.1f0),
    adhesion_matrix   = (
        (0.0f0, 0.0f0, 0.0f0, 0.0f0),    # dead
        (0.0f0, 1.20f0, 0.70f0, 0.60f0), # blue prefers blue, modest to others
        (0.0f0, 0.70f0, 1.25f0, 0.70f0), # white cohesive, neutral to others
        (0.0f0, 0.60f0, 0.70f0, 1.20f0)  # red prefers red
    ),
    polarity_align = 0.55f0,   # push polarity toward neighbors faster
    polarity_decay = 0.00f0,   # keep polarity amplitude longer
    polarity_thresholds = (-0.05f0, 0.05f0), # easier to classify blue/red
)

# ------------------------------------------------------------
# Domaine / dynamique
# ------------------------------------------------------------
const N_CELLS_INITIAL = 500
const MAX_CELLS = N_CELLS_INITIAL
const CELL_RADIUS = 0.15f0
const DT = 0.02f0
const DAMPING = 0.9f0
const MAX_VELOCITY = 1.2f0
const BASE_REPULSION = 1.4f0
const BASE_ADHESION = 0.55f0
const POLARITY_RADIUS = 3.5f0 * CELL_RADIUS
const MPC_GAIN = 1.0f0
const DOMAIN_WIDTH = 9.0f0
const DOMAIN_HEIGHT = 4.0f0
const STEPS = 600
const REPORT_EVERY = 60
const GIF_EVERY = 10           # cadence d'enregistrement des frames
const GIF_FILENAME = "french_flag_no_division.gif"

const DIST_EPS = 1f-6
const FORCE_EPS = 1f-3

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@inline get_collision_weight(t::Int32) = GENOME.collision_weights[Int(t) + 1]
@inline get_adhesion_weight(t::Int32) = GENOME.adhesion_weights[Int(t) + 1]
@inline function adhesion_lookup(t1::Int32, t2::Int32)
    return GENOME.adhesion_matrix[Int(t1) + 1][Int(t2) + 1]
end

@inline function next_polarity(current::Float32, pol_sum::Float32, bias::Float32, count::Int)
    neighbor_pol = count > 0 ? pol_sum / count : current
    neighbor_bias = count > 0 ? bias / count : 0.0f0
    target = clamp(neighbor_pol + 0.5f0 * tanh(neighbor_bias / (POLARITY_RADIUS + 1f0)), -1.0f0, 1.0f0)
    new_pol = (1.0f0 - GENOME.polarity_align - GENOME.polarity_decay) * current + GENOME.polarity_align * target
    return clamp(new_pol, -1.0f0, 1.0f0)
end

# ------------------------------------------------------------
# GPU kernels
# ------------------------------------------------------------
function collision_kernel!(controls, positions, active, n_max, cell_radius)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > n_max || active[idx] != 1
        return
    end
    x = positions[1, idx]; y = positions[2, idx]
    ux = 0.0f0; uy = 0.0f0
    min_dist = 2.2f0 * cell_radius
    min_dist_sq = min_dist * min_dist
    for j in 1:n_max
        if active[j] == 1 && j != idx
            dx = x - positions[1, j]; dy = y - positions[2, j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq && dist_sq > DIST_EPS
                dist = sqrt(dist_sq)
                f = BASE_REPULSION * (min_dist - dist) / (dist + FORCE_EPS)
                ux += f * dx / dist
                uy += f * dy / dist
            end
        end
    end
    controls[1, idx] += ux
    controls[2, idx] += uy
    return
end

function adhesion_kernel!(controls, positions, types, active, n_max, cell_radius)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > n_max || active[idx] != 1
        return
    end
    x = positions[1, idx]; y = positions[2, idx]; t = types[idx]
    ux = 0.0f0; uy = 0.0f0
    range = 4.5f0 * cell_radius
    range_sq = range * range
    for j in 1:n_max
        if active[j] == 1 && j != idx
            dx = x - positions[1, j]; dy = y - positions[2, j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < range_sq && dist_sq > DIST_EPS
                dist = sqrt(dist_sq)
                coef = adhesion_lookup(t, types[j])
                optimal = 3.0f0 * cell_radius
                dev = dist - optimal
                f = -BASE_ADHESION * coef * dev / (dist + FORCE_EPS)
                ux += f * dx / dist
                uy += f * dy / dist
            end
        end
    end
    controls[1, idx] += ux
    controls[2, idx] += uy
    return
end

function polarity_kernel!(polarity, positions, active, n_max, cell_radius)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > n_max || active[idx] != 1
        return
    end
    x = positions[1, idx]; y = positions[2, idx]
    pol_sum = 0.0f0; bias = 0.0f0; count = 0
    r2 = POLARITY_RADIUS * POLARITY_RADIUS
    for j in 1:n_max
        if active[j] == 1 && j != idx
            dx = positions[1, j] - x
            dy = positions[2, j] - y
            dist_sq = dx*dx + dy*dy
            if dist_sq < r2
                pol_sum += polarity[j]
                bias += dx
                count += 1
            end
        end
    end
    polarity[idx] = next_polarity(polarity[idx], pol_sum, bias, count)
    return
end

function fate_kernel!(types, polarity, active, n_max)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > n_max || active[idx] != 1
        return
    end
    p = polarity[idx]
    if p < GENOME.polarity_thresholds[1]
        types[idx] = CELL_TYPE_BLUE
    elseif p > GENOME.polarity_thresholds[2]
        types[idx] = CELL_TYPE_RED
    else
        types[idx] = CELL_TYPE_WHITE
    end
    return
end

function mpc_shape_kernel!(controls, positions, types, active, n_max, domain_w)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > n_max || active[idx] != 1
        return
    end
    t = types[idx]
    target_x = t == CELL_TYPE_BLUE ? -domain_w/3f0 : t == CELL_TYPE_RED ? domain_w/3f0 : 0.0f0
    dx = target_x - positions[1, idx]
    controls[1, idx] += MPC_GAIN * dx
    return
end

function update_kernel!(positions, velocities, controls, active, n_max, dt, damping, vmax, domain_w, domain_h)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx > n_max || active[idx] != 1
        return
    end
    vx = velocities[1, idx] * damping + controls[1, idx] * dt
    vy = velocities[2, idx] * damping + controls[2, idx] * dt
    speed = sqrt(vx*vx + vy*vy)
    if speed > vmax
        vx = vx / speed * vmax
        vy = vy / speed * vmax
    end
    x = positions[1, idx] + vx * dt
    y = positions[2, idx] + vy * dt
    hw = domain_w/2f0; hh = domain_h/2f0
    if x < -hw; x = -hw; vx *= -0.4f0; end
    if x >  hw; x =  hw; vx *= -0.4f0; end
    if y < -hh; y = -hh; vy *= -0.4f0; end
    if y >  hh; y =  hh; vy *= -0.4f0; end
    positions[1, idx] = x; positions[2, idx] = y
    velocities[1, idx] = vx; velocities[2, idx] = vy
    return
end

# ------------------------------------------------------------
# CPU fallbacks (only for quick smoke tests)
# ------------------------------------------------------------
function run_cpu(; record=false, every=GIF_EVERY)
    positions = zeros(Float32, 2, MAX_CELLS)
    velocities = zeros(Float32, 2, MAX_CELLS)
    controls = zeros(Float32, 2, MAX_CELLS)
    types = fill(CELL_TYPE_WHITE, MAX_CELLS)
    polarity = zeros(Float32, MAX_CELLS)
    active = zeros(Int32, MAX_CELLS)
    frames = record ? Vector{NamedTuple{(:positions, :types, :active)}}() : nothing

    spacing = DOMAIN_WIDTH * 0.6f0 / (N_CELLS_INITIAL - 1)
    start_x = -spacing * (N_CELLS_INITIAL - 1) / 2
    for i in 1:N_CELLS_INITIAL
        positions[1, i] = start_x + (i - 1) * spacing
        positions[2, i] = (rand(Float32) - 0.5f0) * CELL_RADIUS
        active[i] = 1
    end
    # organisateurs polarité
    polarity[N_CELLS_INITIAL ÷ 2] = -1.0f0
    polarity[N_CELLS_INITIAL ÷ 2 + 1] = 1.0f0

    if record
        push!(frames, (positions=copy(positions), types=copy(types), active=copy(active)))
    end

    for step in 1:STEPS
        fill!(controls, 0.0f0)
        # collision
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            x = positions[1, i]; y = positions[2, i]
            ux = 0.0f0; uy = 0.0f0
            min_dist = 2.2f0 * CELL_RADIUS
            m2 = min_dist * min_dist
            for j in 1:MAX_CELLS
                if active[j] != 1 || i == j; continue; end
                dx = x - positions[1, j]; dy = y - positions[2, j]
                d2 = dx*dx + dy*dy
                if d2 < m2 && d2 > DIST_EPS
                    d = sqrt(d2)
                    f = BASE_REPULSION * (min_dist - d) / (d + FORCE_EPS)
                    ux += f * dx / d; uy += f * dy / d
                end
            end
            controls[1, i] += ux; controls[2, i] += uy
        end
        # adhesion
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            x = positions[1, i]; y = positions[2, i]; t = types[i]
            ux = 0.0f0; uy = 0.0f0
            rng = 4.5f0 * CELL_RADIUS; r2 = rng * rng
            for j in 1:MAX_CELLS
                if active[j] != 1 || i == j; continue; end
                dx = x - positions[1, j]; dy = y - positions[2, j]
                d2 = dx*dx + dy*dy
                if d2 < r2 && d2 > DIST_EPS
                    d = sqrt(d2)
                    coef = adhesion_lookup(t, types[j])
                    opt = 3.0f0 * CELL_RADIUS
                    dev = d - opt
                    f = -BASE_ADHESION * coef * dev / (d + FORCE_EPS)
                    ux += f * dx / d; uy += f * dy / d
                end
            end
            controls[1, i] += ux; controls[2, i] += uy
        end
        # MPC shape (horizon 1)
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            t = types[i]
            target_x = t == CELL_TYPE_BLUE ? -DOMAIN_WIDTH/3f0 : t == CELL_TYPE_RED ? DOMAIN_WIDTH/3f0 : 0.0f0
            controls[1, i] += MPC_GAIN * (target_x - positions[1, i])
        end
        # polarity
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            x = positions[1, i]; y = positions[2, i]
            ps = 0.0f0; b = 0.0f0; c = 0
            r2 = POLARITY_RADIUS * POLARITY_RADIUS
            for j in 1:MAX_CELLS
                if active[j] != 1 || i == j; continue; end
                dx = positions[1, j] - x; dy = positions[2, j] - y
                d2 = dx*dx + dy*dy
                if d2 < r2
                    ps += polarity[j]; b += dx; c += 1
                end
            end
            polarity[i] = next_polarity(polarity[i], ps, b, c)
        end
        # fate
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            p = polarity[i]
            if p < GENOME.polarity_thresholds[1]
                types[i] = CELL_TYPE_BLUE
            elseif p > GENOME.polarity_thresholds[2]
                types[i] = CELL_TYPE_RED
            else
                types[i] = CELL_TYPE_WHITE
            end
        end
        # dynamics
        hw = DOMAIN_WIDTH/2; hh = DOMAIN_HEIGHT/2
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            vx = velocities[1, i] * DAMPING + controls[1, i] * DT
            vy = velocities[2, i] * DAMPING + controls[2, i] * DT
            s = sqrt(vx*vx + vy*vy)
            if s > MAX_VELOCITY
                vx = vx / s * MAX_VELOCITY; vy = vy / s * MAX_VELOCITY
            end
            x = positions[1, i] + vx * DT
            y = positions[2, i] + vy * DT
            if x < -hw; x = -hw; vx *= -0.4f0; end
            if x >  hw; x =  hw; vx *= -0.4f0; end
            if y < -hh; y = -hh; vy *= -0.4f0; end
            if y >  hh; y =  hh; vy *= -0.4f0; end
            positions[1, i] = x; positions[2, i] = y
            velocities[1, i] = vx; velocities[2, i] = vy
        end
        if step % REPORT_EVERY == 0
            act = findall(==(1), active)
            tmp = types[act]; pol = polarity[act]
            nB = count(==(CELL_TYPE_BLUE), tmp)
            nW = count(==(CELL_TYPE_WHITE), tmp)
            nR = count(==(CELL_TYPE_RED), tmp)
            @printf("Step %-4d | mean pol %.3f | B:%d W:%d R:%d\n", step, mean(pol), nB, nW, nR)
        end
        if record && step % every == 0
            push!(frames, (positions=copy(positions), types=copy(types), active=copy(active)))
        end
    end
    return (positions=positions, types=types, polarity=polarity, active=active, frames=frames)
end

function run_gpu(; record=false, every=GIF_EVERY)
    positions = zeros(Float32, 2, MAX_CELLS)
    velocities = zeros(Float32, 2, MAX_CELLS)
    controls = zeros(Float32, 2, MAX_CELLS)
    types = fill(CELL_TYPE_WHITE, MAX_CELLS)
    polarity = zeros(Float32, MAX_CELLS)
    active = zeros(Int32, MAX_CELLS)
    frames = record ? Vector{NamedTuple{(:positions, :types, :active)}}() : nothing

    spacing = DOMAIN_WIDTH * 0.6f0 / (N_CELLS_INITIAL - 1)
    start_x = -spacing * (N_CELLS_INITIAL - 1) / 2
    for i in 1:N_CELLS_INITIAL
        positions[1, i] = start_x + (i - 1) * spacing
        positions[2, i] = (rand(Float32) - 0.5f0) * CELL_RADIUS
        active[i] = 1
    end
    polarity[N_CELLS_INITIAL ÷ 2] = -1.0f0
    polarity[N_CELLS_INITIAL ÷ 2 + 1] = 1.0f0

    pos_d = CuArray(positions); vel_d = CuArray(velocities); ctrl_d = CuArray(controls)
    types_d = CuArray(types); pol_d = CuArray(polarity); act_d = CuArray(active)
    threads = 256; blocks = cld(MAX_CELLS, threads)

    if record
        push!(frames, (positions=Array(pos_d), types=Array(types_d), active=Array(act_d)))
    end

    for step in 1:STEPS
        fill!(ctrl_d, 0.0f0)
        @cuda threads=threads blocks=blocks collision_kernel!(ctrl_d, pos_d, act_d, MAX_CELLS, CELL_RADIUS)
        @cuda threads=threads blocks=blocks adhesion_kernel!(ctrl_d, pos_d, types_d, act_d, MAX_CELLS, CELL_RADIUS)
        @cuda threads=threads blocks=blocks mpc_shape_kernel!(ctrl_d, pos_d, types_d, act_d, MAX_CELLS, DOMAIN_WIDTH)
        @cuda threads=threads blocks=blocks polarity_kernel!(pol_d, pos_d, act_d, MAX_CELLS, CELL_RADIUS)
        @cuda threads=threads blocks=blocks fate_kernel!(types_d, pol_d, act_d, MAX_CELLS)
        @cuda threads=threads blocks=blocks update_kernel!(pos_d, vel_d, ctrl_d, act_d, MAX_CELLS,
                                                           DT, DAMPING, MAX_VELOCITY, DOMAIN_WIDTH, DOMAIN_HEIGHT)
        if step % REPORT_EVERY == 0
            CUDA.synchronize()
            tmp_types = Array(types_d)[Array(act_d) .== 1]
            tmp_pol = Array(pol_d)[Array(act_d) .== 1]
            nB = count(==(CELL_TYPE_BLUE), tmp_types)
            nW = count(==(CELL_TYPE_WHITE), tmp_types)
            nR = count(==(CELL_TYPE_RED), tmp_types)
            @printf("Step %-4d | mean pol %.3f | B:%d W:%d R:%d\n", step, mean(tmp_pol), nB, nW, nR)
        end
        if record && step % every == 0
            CUDA.synchronize()
            push!(frames, (positions=Array(pos_d), types=Array(types_d), active=Array(act_d)))
        end
    end
    CUDA.synchronize()
    return (positions=Array(pos_d), types=Array(types_d), polarity=Array(pol_d), active=Array(act_d), frames=frames)
end

# ------------------------------------------------------------
# Entrée / sortie
# ------------------------------------------------------------
function cell_color(t)
    return t == CELL_TYPE_BLUE ? :red : t == CELL_TYPE_RED ? :blue : t == CELL_TYPE_WHITE ? :white : :gray
end

function save_plot(res; path="french_flag_no_division.png")
    act = findall(==(1), res.active)
    colors = map(i -> cell_color(res.types[i]), act)
    p = scatter(res.positions[1, act], res.positions[2, act];
        aspect_ratio=:equal, legend=false, background_color=:lightgray,
        xlabel="X", ylabel="Y", color=colors,
        title="French Flag (polarity + MPC shape)")
    savefig(p, path)
    return path
end

function frame_plot(frame)
    act = findall(==(1), frame.active)
    colors = map(i -> cell_color(frame.types[i]), act)
    scatter(frame.positions[1, act], frame.positions[2, act];
        aspect_ratio=:equal, legend=false, background_color=:lightgray,
        xlabel="X", ylabel="Y", color=colors,
        xlim=(-DOMAIN_WIDTH/2, DOMAIN_WIDTH/2), ylim=(-DOMAIN_HEIGHT/2, DOMAIN_HEIGHT/2),
        title="French Flag (polarity + MPC shape)")
end

function save_gif(frames; path=GIF_FILENAME)
    anim = @animate for fr in frames
        frame_plot(fr)
    end
    gif(anim, path; fps = max(1, Int(cld(length(frames), 10))))
    return path
end

function print_genome()
    println("=== Génome numérique (poids) ===")
    println("Collision weights :", GENOME.collision_weights)
    println("Adhesion weights  :", GENOME.adhesion_weights)
    println("Adhesion matrix (dead, blue, white, red):")
    for row in GENOME.adhesion_matrix
        println("  ", row)
    end
    println("Polarity thresholds (blue/red): ", GENOME.polarity_thresholds)
end

function main()
    Random.seed!(42)
    print_genome()
    use_gpu = CUDA.functional()
    println("GPU disponible : ", use_gpu)
    record_gif = true
    res = use_gpu ? run_gpu(record=record_gif) : run_cpu(record=record_gif)
    img = save_plot(res)
    if record_gif && res.frames !== nothing
        gif_path = save_gif(res.frames)
        println("GIF sauvegardé: $gif_path")
    end
    act = findall(==(1), res.active)
    types = res.types[act]
    nB = count(==(CELL_TYPE_BLUE), types)
    nW = count(==(CELL_TYPE_WHITE), types)
    nR = count(==(CELL_TYPE_RED), types)
    println("\nRépartition finale (actives) :")
    println("  🔵 Bleu : $nB")
    println("  ⚪ Blanc: $nW")
    println("  🔴 Rouge: $nR")
    println("Plot sauvegardé: $img")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

println("faire :\njulia>main()")