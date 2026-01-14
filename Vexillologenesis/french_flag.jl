"""
Minimal, gradient-free French Flag (Blue / White / Red)
=======================================================

Principes (inspiré Fronville/Mercier) :
- Une seule cellule initiale au centre, division stochastique limitée.
- Génome numérique simple : adhésion / répulsion, polarité locale, seuils de différenciation.
- Polarité locale : moyenne des voisins + biais gauche/droite, relaxation.
- MPC de forme (horizon 1) : poussée douce vers centres de bandes, avec rappel vers position initiale pour éviter la dérive.
- Sénescence / apoptose : après plusieurs divisions ou âge élevé, mouvement ralenti et probabilité de mort accrue.
- GPU optionnel : ce script fonctionne sur CPU par défaut; l’essentiel est la logique émergente.

Usage :
    julia --project french_flag_minimal_polarity.jl
Sorties :
    - french_flag_minimal.png (état final)
    - french_flag_minimal.gif  (morphogenèse)
    - Impression du génome
"""

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
    adhesion_weights  = (0.0f0, 0.9f0, 1.0f0, 0.9f0),
    adhesion_matrix   = (
        (0.0f0, 0.0f0, 0.0f0, 0.0f0),   # dead
        (0.0f0, 1.00f0, 0.85f0, 0.70f0), # blue
        (0.0f0, 0.85f0, 1.00f0, 0.85f0), # white
        (0.0f0, 0.70f0, 0.85f0, 1.00f0)  # red
    ),
    polarity_align = 0.35f0,
    polarity_decay = 0.01f0,
    polarity_thresholds = (-0.25f0, 0.25f0), # blue / red
)

# ------------------------------------------------------------
# Domaine / dynamique
# ------------------------------------------------------------
const N_CELLS_INITIAL = 1
const MAX_CELLS = 700
const CELL_RADIUS = 0.15f0
const DT = 0.02f0
const DAMPING = 0.9f0
const MAX_VELOCITY = 1.2f0
const BASE_REPULSION = 1.8f0
const BASE_ADHESION = 0.35f0
const POLARITY_RADIUS = 3.5f0 * CELL_RADIUS
const MPC_GAIN = 0.35f0
const RESTORE_GAIN = 0.05f0
const DOMAIN_WIDTH = 9.0f0
const DOMAIN_HEIGHT = 4.0f0
const STEPS = 1200
const REPORT_EVERY = 60

const MAX_DIVISIONS = 8
const DIV_PROB = 0.35f0
const MIN_DIV_AGE = 6
const SENESCENCE_AGE = 999   # pratiquement pas de sénescence dans l'horizon simulé
const APOPTOSIS_BASE = 0.0005f0
const APOPTOSIS_SENESCENT = 0.002f0

# Cibles de population pour équilibrer les couleurs
const TARGET_PER_COLOR = 200
const THRESH_BIAS_STEP = 0.01f0
const THRESH_MIN = -0.35f0
const THRESH_MAX = 0.35f0

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
# Simulation CPU (division, sénescence, apoptose)
# ------------------------------------------------------------
function run_cpu(; save_history::Bool=true, history_stride::Int=8)
    positions = zeros(Float32, 2, MAX_CELLS)
    velocities = zeros(Float32, 2, MAX_CELLS)
    controls = zeros(Float32, 2, MAX_CELLS)
    types = fill(CELL_TYPE_WHITE, MAX_CELLS)
    polarity = zeros(Float32, MAX_CELLS)
    active = zeros(Int32, MAX_CELLS)
    divisions = zeros(Int32, MAX_CELLS)
    ages = zeros(Int32, MAX_CELLS)
    init_x = zeros(Float32, MAX_CELLS)

    # cellule fondatrice
    positions[:, 1] .= (0.0f0, 0.0f0)
    init_x[1] = 0.0f0
    active[1] = 1
    types[1] = CELL_TYPE_WHITE
    polarity[1] = 0.0f0

    history_positions = save_history ? Vector{Array{Float32,2}}() : nothing
    history_types = save_history ? Vector{Array{Int32,1}}() : nothing
    history_active = save_history ? Vector{Array{Int32,1}}() : nothing

    div_count = 0
    apoptosis_count = 0

    th_low = GENOME.polarity_thresholds[1]
    th_high = GENOME.polarity_thresholds[2]

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
            rng = 4.0f0 * CELL_RADIUS; r2 = rng * rng
            for j in 1:MAX_CELLS
                if active[j] != 1 || i == j; continue; end
                dx = x - positions[1, j]; dy = y - positions[2, j]
                d2 = dx*dx + dy*dy
                if d2 < r2 && d2 > DIST_EPS
                    d = sqrt(d2)
                    coef = adhesion_lookup(t, types[j])
                    opt = 2.6f0 * CELL_RADIUS
                    dev = d - opt
                    f = -BASE_ADHESION * coef * dev / (d + FORCE_EPS)
                    ux += f * dx / d; uy += f * dy / d
                end
            end
            controls[1, i] += ux; controls[2, i] += uy
        end
        # MPC shape + rappel anti-dérive
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            t = types[i]
            target_x = t == CELL_TYPE_BLUE ? -DOMAIN_WIDTH/3f0 : t == CELL_TYPE_RED ? DOMAIN_WIDTH/3f0 : 0.0f0
            controls[1, i] += MPC_GAIN * (target_x - positions[1, i])
            controls[1, i] += RESTORE_GAIN * (init_x[i] - positions[1, i])
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
        # fate (avec seuils adaptatifs)
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            p = polarity[i]
            if p < th_low
                types[i] = CELL_TYPE_BLUE
            elseif p > th_high
                types[i] = CELL_TYPE_RED
            else
                types[i] = CELL_TYPE_WHITE
            end
        end

        # ajustement des seuils pour équilibrer les couleurs
        if step % 4 == 0
            act = findall(==(1), active)
            tmp = types[act]
            nB = count(==(CELL_TYPE_BLUE), tmp)
            nW = count(==(CELL_TYPE_WHITE), tmp)
            nR = count(==(CELL_TYPE_RED), tmp)
            # si bleu trop bas, on élargit la zone bleue (seuil plus proche de 0)
            if nB < TARGET_PER_COLOR
                th_low = min(th_low + THRESH_BIAS_STEP, THRESH_MAX)
            elseif nB > TARGET_PER_COLOR
                th_low = max(th_low - THRESH_BIAS_STEP, THRESH_MIN)
            end
            # si rouge trop bas, on élargit la zone rouge (seuil plus proche de 0)
            if nR < TARGET_PER_COLOR
                th_high = max(th_high - THRESH_BIAS_STEP, THRESH_MIN)
            elseif nR > TARGET_PER_COLOR
                th_high = min(th_high + THRESH_BIAS_STEP, THRESH_MAX)
            end
            # garder th_low < th_high
            if th_low >= th_high
                mid = (th_low + th_high) / 2
                th_low = mid - 0.02f0
                th_high = mid + 0.02f0
            end
        end

        # division
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            if divisions[i] >= MAX_DIVISIONS || ages[i] < MIN_DIV_AGE
                continue
            end
            if rand() < DIV_PROB && sum(active) < MAX_CELLS - 1
                new_idx = findfirst(x -> x == 0, active)
                if new_idx !== nothing
                    angle = 2f0 * pi * rand(Float32)
                    r = CELL_RADIUS * 1.1f0
                    positions[1, new_idx] = positions[1, i] + r * cos(angle)
                    positions[2, new_idx] = positions[2, i] + r * sin(angle)
                    init_x[new_idx] = positions[1, new_idx]
                    velocities[:, new_idx] .= 0
                    controls[:, new_idx] .= 0
                    polarity[new_idx] = polarity[i]
                    types[new_idx] = types[i]
                    active[new_idx] = 1
                    divisions[i] += 1
                    ages[new_idx] = 0
                    div_count += 1
                end
            end
        end

        # apoptose / sénescence
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            ages[i] += 1
            prob = APOPTOSIS_BASE
            if ages[i] > SENESCENCE_AGE || divisions[i] >= MAX_DIVISIONS
                prob += APOPTOSIS_SENESCENT
            end
            if rand() < prob
                active[i] = 0
                apoptosis_count += 1
                continue
            end
        end

        # dynamics
        hw = DOMAIN_WIDTH/2; hh = DOMAIN_HEIGHT/2
        for i in 1:MAX_CELLS
            if active[i] != 1; continue; end
            damp = (ages[i] > SENESCENCE_AGE || divisions[i] >= MAX_DIVISIONS) ? DAMPING * 0.8f0 : DAMPING
            vx = velocities[1, i] * damp + controls[1, i] * DT
            vy = velocities[2, i] * damp + controls[2, i] * DT
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

        if save_history && step % history_stride == 0
            push!(history_positions, copy(positions))
            push!(history_types, copy(types))
            push!(history_active, copy(active))
        end
        if step % REPORT_EVERY == 0
            act = findall(==(1), active)
            tmp = types[act]; pol = polarity[act]
            nB = count(==(CELL_TYPE_BLUE), tmp)
            nW = count(==(CELL_TYPE_WHITE), tmp)
            nR = count(==(CELL_TYPE_RED), tmp)
            @printf("Step %-4d | mean pol %.3f | B:%d W:%d R:%d | active %d\n",
                    step, mean(pol), nB, nW, nR, length(act))
        end
    end
    return (
        positions=positions,
        types=types,
        polarity=polarity,
        active=active,
        history_positions=history_positions,
        history_types=history_types,
        history_active=history_active,
        divisions_total=div_count,
        apoptosis_total=apoptosis_count,
    )
end

# ------------------------------------------------------------
# Entrée / sortie
# ------------------------------------------------------------
function save_plot(res; path="french_flag_minimal.png")
    act = findall(==(1), res.active)
    colors = map(act) do i
        t = res.types[i]
        t == CELL_TYPE_BLUE ? :red : t == CELL_TYPE_RED ? :blue : t == CELL_TYPE_WHITE ? :white : :gray
    end
    title_str = "Drapeau français (division + polarité)  +$(res.divisions_total)/-$(res.apoptosis_total)"
    p = scatter(res.positions[1, act], res.positions[2, act];
        aspect_ratio=:equal, legend=false, background_color=:lightgray,
        xlabel="X", ylabel="Y", color=colors,
        title=title_str)
    savefig(p, path)
    return path
end

function save_gif(res; path="french_flag_minimal.gif")
    if res.history_positions === nothing
        return nothing
    end
    anim = @animate for (pos, typ, actv) in zip(res.history_positions, res.history_types, res.history_active)
        act = findall(==(1), actv)
        cols = map(act) do i
            t = typ[i]
            t == CELL_TYPE_BLUE ? :red : t == CELL_TYPE_RED ? :blue : t == CELL_TYPE_WHITE ? :white : :gray
        end
        scatter(pos[1, act], pos[2, act];
            aspect_ratio=:equal, legend=false, background_color=:lightgray,
            xlabel="X", ylabel="Y", color=cols,
            xlim=(-DOMAIN_WIDTH/2, DOMAIN_WIDTH/2), ylim=(-DOMAIN_HEIGHT/2, DOMAIN_HEIGHT/2),
            title="Morphogenèse")
    end
    gif(anim, path; fps=12)
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
    println("GPU : non requis pour cette version (CPU focalisé sur logique émergente).")
    res = run_cpu(save_history=true, history_stride=6)
    img = save_plot(res)
    gif_path = save_gif(res)
    act = findall(==(1), res.active)
    types = res.types[act]
    nB = count(==(CELL_TYPE_BLUE), types)
    nW = count(==(CELL_TYPE_WHITE), types)
    nR = count(==(CELL_TYPE_RED), types)
    println("\nRépartition finale (actives) :")
    println("  🔵 Bleu : $nB")
    println("  ⚪ Blanc: $nW")
    println("  🔴 Rouge: $nR")
    println("  + Divisions : $(res.divisions_total)")
    println("  - Apoptoses : $(res.apoptosis_total)")
    println("Plot sauvegardé: $img")
    if gif_path !== nothing
        println("GIF sauvegardé: $gif_path")
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end