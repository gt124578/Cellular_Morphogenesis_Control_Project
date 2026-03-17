using DifferentialEquations, LinearAlgebra, Plots, Statistics

# ==============================================================================
# DRAPEAU FRANÇAIS WOLPERT
# ==============================================================================
# Basé sur meca_flu/morpho_flu_test.jl (Turing → Wolpert)
#
# Changements par rapport au modèle de Turing :
#   1. Briser la symétrie : variable Org (Organisateur) sur le bord gauche
#   2. Chimie Wolpert : dA = Da*Δ(A) - degrad*A + prod*Org  (pas de Gray-Scott)
#   3. Mécanique anisotrope : axe j (horizontal) pousse plus fort → élongation
#   4. Advection de Org : transporté par le fluide comme ρ et A
#   5. Décodage génétique : seuils sur A → Bleu | Blanc | Rouge
#   6. Homéostasie fibroblastes : même règle de bord que le code de base
# ==============================================================================

const N  = 60
const L  = 12.0
const dx = L / N

# ------------------------------------------------------------------------------
# PARAMÈTRES
# [1:Diff_Rho, 2:Croissance, 3:Diff_A, 4:Dégradation_A, 5:Production_A,
#  6:Pression_K, 7:Visco_Base, 8:Force_Active, 9:Anisotropie]
# ------------------------------------------------------------------------------
const P_DRAPEAU = [
    0.02,   # Diffusion Densité     (relaxation mécanique)
    0.6,    # Taux de Croissance    (division cellulaire)
    1.0,    # Diffusion Morphogène  (rapide → gradient lisse sur tout le tissu)
    0.05,   # Dégradation A         (λ = sqrt(Da/degrad) ≈ 4.5 unités)
    0.05,   # Production de A       (par l'Organisateur : A0 = prod/degrad = 1.0)
    2.0,    # Raideur Pression      (incompressibilité)
    0.5,    # Viscosité Fluide
    0.8,    # Force Active          (chimie → poussée mécanique)
    2.5,    # Anisotropie           (axe horizontal pousse 2.5× plus fort → rectangle)
]

# Seuils de décodage génétique (Wolpert)
# Ajustez si nécessaire selon la taille finale du tissu
const SEUIL_HAUT = 0.40   # A > 0.40 → Bleu  (proche du pôle organisateur)
const SEUIL_BAS  = 0.15   # A < 0.15 → Rouge (loin du pôle organisateur)

# Seuil minimal de range pour activer les seuils adaptatifs (évite les divisions par ~0)
const MIN_GRADIENT_RANGE = 0.01

# Multiplicateur de viscosité des fibroblastes (simule la rigidification du cytosquelette)
const FIBROBLAST_VISCOSITY_FACTOR = 50.0

# ==============================================================================
# HELPERS NUMÉRIQUES
# ==============================================================================

@inline function sigmoid(x)
    return 1.0 / (1.0 + exp(-15.0 * (x - 0.5)))
end

# Advection upwind 1er ordre (schéma conservatif)
@inline function upwind_advection(M, vx, vy, i, j, dx)
    flux_x = (vx > 0) ? vx * (M[i,j] - M[i-1,j]) : vx * (M[i+1,j] - M[i,j])
    flux_y = (vy > 0) ? vy * (M[i,j] - M[i,j-1]) : vy * (M[i,j+1] - M[i,j])
    return -(flux_x + flux_y) / dx
end

# ==============================================================================
# MOTEUR : DRAPEAU WOLPERT (INFORMATION POSITIONNELLE)
# ==============================================================================
# État : [Rho, A, Org, F]  (4 × N² variables)
#   Rho : densité cellulaire
#   A   : morphogène (gradient Wolpert)
#   Org : Gène Organisateur (pôle gauche, advecté par le fluide)
#   F   : Fibroblastes (coque de stabilisation)
# ==============================================================================
function drapeau_morphogenesis!(du, u, p, t)
    D_rho, k_growth, Da, degrad_a, prod_a, p_stiff, visc_base, k_active, aniso = p

    # --- A. DÉBALLAGE DES VARIABLES D'ÉTAT ---
    sz   = N^2
    Rho  = reshape(@view(u[1:sz]),           N, N)
    A    = reshape(@view(u[sz+1:2*sz]),      N, N)
    Org  = reshape(@view(u[2*sz+1:3*sz]),    N, N)
    F    = reshape(@view(u[3*sz+1:4*sz]),    N, N)

    dRho = reshape(@view(du[1:sz]),          N, N)
    dA   = reshape(@view(du[sz+1:2*sz]),     N, N)
    dOrg = reshape(@view(du[2*sz+1:3*sz]),   N, N)
    dF   = reshape(@view(du[3*sz+1:4*sz]),   N, N)

    # --- B. MÉCANIQUE DES FLUIDES ACTIFS (ANISOTROPE) ---
    # VX : vitesse dans la direction i (lignes = vertical dans le heatmap)
    # VY : vitesse dans la direction j (colonnes = horizontal dans le heatmap)
    # → L'anisotropie sur VY crée l'élongation horizontale (antéro-postérieure)
    VX = zeros(Float64, N, N)
    VY = zeros(Float64, N, N)

    @inbounds for j in 2:N-1, i in 2:N-1
        # Viscosité effective : les fibroblastes figent le mouvement (×FIBROBLAST_VISCOSITY_FACTOR)
        visc = visc_base * (1.0 + FIBROBLAST_VISCOSITY_FACTOR * F[i,j])

        # Potentiel total = mécanique (pression stérique) + actif (chimie)
        # Calculé pour chaque voisin (évite le bug du code de base avec scalaire indexé)
        pot(ri, rj) = p_stiff * max(0.0, Rho[ri,rj] - 1.0)^2 +
                      k_active * A[ri,rj] * Rho[ri,rj]

        grad_x = (pot(i+1,j) - pot(i-1,j)) / (2dx)   # direction i (vertical)
        grad_y = (pot(i,j+1) - pot(i,j-1)) / (2dx)   # direction j (horizontal)

        # ANISOTROPIE : axe horizontal (j) pousse `aniso` fois plus fort
        # → le tissu s'étire en rectangle plutôt qu'en cercle
        VX[i,j] = -tanh(grad_x       / visc)
        VY[i,j] = -tanh(aniso * grad_y / visc)
    end

    # --- C. DYNAMIQUE DE LA MATIÈRE ---
    @inbounds for j in 2:N-1, i in 2:N-1
        # Masque tissu (0 dans le vide, 1 dans le tissu)
        mask = sigmoid(Rho[i,j])

        # 1. Transport (advection Upwind)
        adv_rho = upwind_advection(Rho, VX[i,j], VY[i,j], i, j, dx)
        adv_a   = upwind_advection(A,   VX[i,j], VY[i,j], i, j, dx)
        adv_org = upwind_advection(Org, VX[i,j], VY[i,j], i, j, dx)
        adv_f   = upwind_advection(F,   VX[i,j], VY[i,j], i, j, dx)

        # 2. Diffusion (Laplaciens discrets)
        lap_rho = (Rho[i+1,j] + Rho[i-1,j] + Rho[i,j+1] + Rho[i,j-1] - 4*Rho[i,j]) / dx^2
        lap_a   = (A[i+1,j]   + A[i-1,j]   + A[i,j+1]   + A[i,j-1]   - 4*A[i,j])   / dx^2

        # 3. Réactions biologiques

        # Croissance logistique (inhibée par les fibroblastes et la surpopulation)
        growth = k_growth * Rho[i,j] * (1.2 - Rho[i,j]) * (1.0 - F[i,j])

        # WOLPERT : Diffusion + Dégradation + Source à l'Organisateur
        # (remplace Gray-Scott : plus de terme A²B)
        # dA/dt = Da·Δ(A) - degrad·A + prod·Org
        reac_a = -degrad_a * A[i,j] + prod_a * Org[i,j]

        # Différenciation Fibroblastes : aux bords (fort gradient de densité)
        # → crée une coque solide qui arrête la croissance (homéostasie)
        gradient_rho = sqrt((Rho[i+1,j]-Rho[i-1,j])^2 + (Rho[i,j+1]-Rho[i,j-1])^2)
        rate_fibro = 0.0
        if Rho[i,j] > 0.2 && gradient_rho > 0.3
            rate_fibro = 0.1 * (1.0 - F[i,j])
        end

        # --- D. ASSEMBLAGE ---
        dRho[i,j] = adv_rho + (D_rho * lap_rho) + growth
        dA[i,j]   = mask * (Da * lap_a + reac_a) + adv_a
        # ADVECTION DE L'ORGANISATEUR : le pôle gauche reste gauche même en croissance
        dOrg[i,j] = adv_org
        dF[i,j]   = adv_f + rate_fibro
    end

    # Conditions aux limites miroirs (bords de la boîte)
    @views begin
        dRho[1,:].=0; dRho[N,:].=0; dRho[:,1].=0; dRho[:,N].=0
        dA[1,:].=0;   dA[N,:].=0;   dA[:,1].=0;   dA[:,N].=0
        dOrg[1,:].=0; dOrg[N,:].=0; dOrg[:,1].=0; dOrg[:,N].=0
        dF[1,:].=0;   dF[N,:].=0;   dF[:,1].=0;   dF[:,N].=0
    end
end

# ==============================================================================
# INITIALISATION (BRISER LA SYMÉTRIE → CRÉER UN "PÔLE")
# ==============================================================================
u0   = zeros(Float64, 4*N*N)
Rho0 = reshape(@view(u0[1:N^2]),           N, N)
A0   = reshape(@view(u0[N^2+1:2*N^2]),     N, N)
Org0 = reshape(@view(u0[2*N^2+1:3*N^2]),   N, N)

mid    = N ÷ 2
radius = 5  # rayon initial en cellules

for i in 1:N, j in 1:N
    dist = sqrt((i - mid)^2 + (j - mid)^2)
    if dist < radius
        Rho0[i,j] = 1.0
        A0[i,j]   = 0.01 * rand()   # faible bruit (le gradient sera imposé par Org)

        # CENTRE ORGANISATEUR : bord gauche de la goutte initiale
        # j < mid → demi-gauche de la goutte → Org = 1.0
        # Le morphogène sera produit ici, diffusera vers la droite et se dégradera
        if j < mid
            Org0[i,j] = 1.0
        end
    end
end

# ==============================================================================
# RÉSOLUTION
# ==============================================================================
println("Modèle Drapeau Français — Wolpert (Information Positionnelle)")
println("Variables d'état : Rho | A (morphogène) | Org (organisateur) | F (fibroblastes)")
println("Chimie : dA = Da·Δ(A) - degrad·A + prod·Org  (pas de Turing)")
println("Mécanique : anisotropie j × $(P_DRAPEAU[9]) → élongation horizontale")
println()

prob = ODEProblem(drapeau_morphogenesis!, u0, (0.0, 600.0), P_DRAPEAU)

println("Résolution (Tsit5)...")
sol = solve(prob, Tsit5(), saveat=2.0, progress=true)
println("Résolution terminée. $(length(sol.t)) pas sauvegardés.")

# ==============================================================================
# VISUALISATION : DÉCODAGE GÉNÉTIQUE (WOLPERT) + DRAPEAU
# ==============================================================================
println("Création du GIF...")

anim = @animate for t_idx in 1:length(sol.t)
    u  = sol.u[t_idx]
    sz = N^2
    Rho = reshape(u[1:sz],          N, N)
    A   = reshape(u[sz+1:2*sz],     N, N)
    Org = reshape(u[2*sz+1:3*sz],   N, N)
    F   = reshape(u[3*sz+1:4*sz],   N, N)

    # ---- Décodage Génétique (Wolpert) ----
    # Les cellules "lisent" la concentration de A et choisissent leur phénotype.
    # Utilise des seuils adaptatifs si le gradient est établi, sinon les seuils fixes.
    a_vals = Float64[]
    for i in 1:N, j in 1:N
        if Rho[i,j] > 0.1
            push!(a_vals, A[i,j])
        end
    end

    if length(a_vals) > 10
        a_min = minimum(a_vals)
        a_max = maximum(a_vals)
        a_range = a_max - a_min
        # Seuils adaptatifs : divise le gradient en 3 bandes égales
        seuil_h = (a_range > MIN_GRADIENT_RANGE) ? (a_min + a_range * 2.0/3.0) : SEUIL_HAUT
        seuil_l = (a_range > MIN_GRADIENT_RANGE) ? (a_min + a_range * 1.0/3.0) : SEUIL_BAS
    else
        seuil_h = SEUIL_HAUT
        seuil_l = SEUIL_BAS
    end

    # Carte couleur : 1.0 = Bleu, 0.5 = Blanc, 0.0 = Rouge, NaN = Vide
    flag_image = fill(NaN, N, N)
    for i in 1:N, j in 1:N
        if Rho[i,j] > 0.1
            if A[i,j] > seuil_h
                flag_image[i,j] = 1.0    # Bleu (fort A, proche organisateur)
            elseif A[i,j] > seuil_l
                flag_image[i,j] = 0.5    # Blanc (A moyen)
            else
                flag_image[i,j] = 0.0    # Rouge (faible A, loin organisateur)
            end
        end
    end

    t_str = string(round(Int, sol.t[t_idx]))

    p1 = heatmap(Rho, c=:grays,
                 title="Densité (t=$t_str)", clims=(0, 1.2), axis=false,
                 colorbar=false)

    a_clim = max(maximum(A), 0.001)
    p2 = heatmap(A, c=:plasma,
                 title="Morphogène A (gradient Wolpert)", clims=(0, a_clim),
                 axis=false, colorbar=true)

    # Drapeau : Bleu | Blanc | Rouge de gauche (pôle Org, fort A) à droite (faible A)
    # Encodage : flag_image 1.0=Bleu, 0.5=Blanc, 0.0=Rouge
    # cgrad([:red, :white, :blue]) : valeur 0.0→rouge, 0.5→blanc, 1.0→bleu (correct)
    p3 = heatmap(flag_image,
                 c=cgrad([:red, :white, :blue]),
                 title="Drapeau Français (Bleu|Blanc|Rouge)",
                 clims=(0, 1), axis=false, colorbar=false,
                 background_color_inside=:lightgray)

    p4 = heatmap(F, c=:algae,
                 title="Fibroblastes (coque de stabilisation)",
                 clims=(0, 1), axis=false, colorbar=false)

    plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800),
         plot_title="Drapeau Français (Wolpert) — t=$t_str")
end

gif(anim, "morphogenese_active.gif", fps=15)
println("Terminé. Drapeau sauvegardé : morphogenese_active.gif")