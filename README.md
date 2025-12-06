# Contrôle de la Morphogenèse Cellulaire via un Système de Commande

Ce projet vise à concevoir et implémenter un système de contrôle en Julia capable de guider l'auto-organisation d'une population de cellules vers une forme géométrique cible prédéfinie.

---

## Table des Matières
1.  [Contexte Scientifique](#1-contexte-scientifique)
    *   [La Morphogenèse](#la-morphogenèse)
    *   [Les Équations Mutationnelles](#les-équations-mutationnelles)
    *   [La Théorie du Contrôle](#la-théorie-du-contrôle)
2.  [Objectifs du Projet](#2-objectifs-du-projet)
3.  [Dépendances Clés](#dépendances-clés)
4.  [Architecture du Contrôleur](#4-architecture-du-contrôleur)
5.  [Démarrage Rapide](#5-démarrage-rapide)
6.  [Structure du Dépôt](#6-structure-du-dépôt)
7.  [Contributeurs](#7-contributeurs)

---

### 1. Contexte Scientifique

#### La Morphogenèse
La morphogenèse est le processus biologique par lequel un organisme développe sa forme. Ce projet s'attaque à une question fondamentale de ce domaine : comment des interactions et des règles locales simples entre cellules peuvent-elles aboutir à l'émergence d'une structure globale complexe et cohérente ?

#### Les Équations Mutationnelles
Pour modéliser l'évolution de notre population de cellules, nous utilisons le formalisme mathématique des **équations mutationnelles**. Ce cadre est particulièrement adapté à notre problème car il gère nativement le défi principal de la simulation : la **dimensionnalité variable de l'espace d'états**. À chaque division ou mort cellulaire, le nombre de cellules change, et donc la taille du vecteur d'état qui décrit le système. Les équations mutationnelles nous fournissent les outils pour décrire cette dynamique de manière rigoureuse.

#### La Théorie du Contrôle
Nous appliquons les principes de la théorie du contrôle pour guider le système. Notre contrôleur agit comme un analogue de l'environnement biologique (la matrice extracellulaire, les gradients de facteurs de croissance), qui régule le comportement cellulaire. En observant l'état actuel du système et en le comparant à la forme désirée, le contrôleur ajuste les "paramètres environnementaux" pour corriger la trajectoire de la morphogenèse.

### 2. Objectifs du Projet

L'objectif principal est de développer un système de contrôle en boucle fermée robuste pour la morphogenèse. Les sous-objectifs sont :

-   **Intégrer** le simulateur de morphogenèse existant avec un module de contrôle.
-   **Généraliser** le concept de contrôle 1D à un cadre multi-dimensionnel pour des formes complexes.
-   **Implémenter** une boucle de contrôle (Mesure → Comparaison → Décision → Action).
-   **Définir** une métrique d'erreur pertinente (Distance de Hausdorff) pour quantifier l'écart entre la forme actuelle et la forme cible.
-   **Tester** la performance du système sur des formes de complexité croissante.
-   **Évaluer** la robustesse du contrôleur face à des perturbations stochastiques (ex: mort cellulaire aléatoire).


### 3. Dépendances Clés
Ce projet s'appuie sur deux dépôts préexistants :
1.  [![GitHub Repo](https://img.shields.io/badge/GitHub-ShapeGrowthModule-blue)](https://github.com/afronvil/ShapeGrowthModule)

2. [![GitHub Repo](https://img.shields.io/badge/GitHub-OptimalControl-blue)](https://github.com/control-toolbox/OptimalControl.jl)


### 4. Architecture du Contrôleur

Le système est une boucle de rétroaction qui fonctionne de manière itérative :

1.  **Mesure :** À un instant `t`, le contrôleur extrait l'état actuel du système `K(t)` depuis le simulateur. Cet état est l'ensemble des coordonnées de toutes les cellules vivantes.
    `K(t) = {c₁, c₂, ..., cₙ(t)}` où `cᵢ` est le vecteur de coordonnées de la cellule `i`.

2.  **Comparaison :** Le contrôleur calcule une erreur `e(t)` en mesurant la "distance" entre l'état `K(t)` et la forme cible `K_cible` à l'aide d'une métrique.
    `e(t) = Hausdorff(K(t), K_cible)`

3.  **Décision :** Sur la base de l'erreur `e(t)`, l'algorithme de contrôle (la "loi de commande") calcule une nouvelle commande `u(t)`. Cette commande est un vecteur de paramètres qui pilotent l'environnement.
    `u(t) = f(e(t))`

4.  **Action :** La commande `u(t)` est appliquée au simulateur, modifiant les conditions environnementales pour l'itération suivante.

Le cycle recommence, réduisant progressivement l'erreur `e(t)` jusqu'à ce que la forme `K(t)` converge vers `K_cible`.

### 5. Démarrage Rapide

Instructions pour installer et lancer.

1.  **Prérequis :**
    -   [Julia](https://julialang.org/downloads/) (version 1.x).
    -   Git.

2.  **Installation :**
    ```bash
    # Cloner le dépôt
    git clone https://github.com/gt124578/Cellular_Morphogenesis_Control_Project/
    cd Cellular_Morphogenesis_Control_Project

    # Lancer Julia
    julia

    # Activer l'environnement du projet et installer les dépendances
    julia> ]
    pkg> activate .
    pkg> instantiate
    ```

3.  **Lancer une simulation :**
    
    **🚀 NOUVEAU : Simulation GPU Agent-Based (RECOMMANDÉ pour n≥100) :**
    ```bash
    # Test rapide avec 50 cellules (~5-10 secondes sur GPU)
    julia --project=. gpu_agent_test.jl
    
    # Simulation standard avec 100 cellules (~10-20 secondes)
    julia --project=. gpu_agent_n100.jl
    
    # Grande échelle avec 500 cellules (~30-60 secondes)
    julia --project=. gpu_agent_n500.jl
    
    # Très grande échelle avec 1000 cellules (~1-2 minutes)
    julia --project=. gpu_agent_n1000.jl
    ```
    
    **Simulation avec contrôle optimal (pour petites échelles n≤30) :**
    ```bash
    # Test rapide avec 10 cellules (~1-2 minutes)
    julia --project=. test_morphogenesis_quick.jl
    
    # Simulation avec 50 cellules (~10-30 minutes, CPU)
    julia --project=. morphogenesis_n50.jl
    ```
    
    **Simulations existantes :**
    ```julia
    # Exécuter les scripts de test originaux
    cd test_optimal_control
    julia --project=.. morphogénèse_R6.jl
    julia --project=.. morphogenese_R21.jl
    ```

### 6. Structure du Dépôt

```
.
├── morphogenesis_gpu_agent.jl          # 🚀 Simulation GPU agent-based (NOUVEAU)
├── gpu_agent_test.jl                   # Test GPU rapide (50 cellules)
├── gpu_agent_n100.jl                   # GPU 100 cellules
├── gpu_agent_n500.jl                   # GPU 500 cellules
├── gpu_agent_n1000.jl                  # GPU 1000 cellules
├── GPU_AGENT_README.md                 # Documentation GPU détaillée
├── morphogenesis_oxygen_gpu.jl         # Simulation optimal control (CPU)
├── morphogenesis_n50.jl                # Optimal control 50 cellules
├── morphogenesis_n100.jl               # Optimal control 100 cellules
├── test_morphogenesis_quick.jl         # Test rapide (10 cellules)
├── OXYGEN_SIMULATION_README.md         # Documentation optimal control
├── RESUME_FR.md                        # Résumé en français
├── Project.toml                        # Dépendances Julia
├── test_optimal_control/               # Scripts de test originaux
│   ├── morphogénèse_R6.jl
│   ├── morphogenese_R21.jl
│   └── ...
└── README.md                           # Ce fichier
```

## Deux Approches de Simulation

### 🚀 GPU Agent-Based (RECOMMANDÉ pour n≥100)

**Caractéristiques :**
- ✅ Vraie accélération GPU avec kernels CUDA
- ✅ Scalable jusqu'à 1000+ cellules
- ✅ Temps de calcul : secondes à minutes
- ✅ Biologiquement réaliste (règles locales)
- ✅ Inspiré de la recherche (Jeannin-Girardon, Ballet, Rodin)

**Performances :**
- 50 cellules : ~5-10 secondes
- 100 cellules : ~10-20 secondes
- 500 cellules : ~30-60 secondes
- 1000 cellules : ~1-2 minutes

**Quand utiliser :**
- Simulations à grande échelle (n>50)
- Besoin de performance
- GPU NVIDIA disponible

Voir [GPU_AGENT_README.md](GPU_AGENT_README.md) pour plus de détails.

### 📊 Optimal Control (Pour petites échelles)

**Caractéristiques :**
- ✅ Trajectoires mathématiquement optimales
- ✅ Contrôle précis avec OptimalControl.jl
- ✅ Contraintes de collision explicites
- ❌ Temps de calcul O(n³) sur CPU
- ❌ Limite pratique : n≤30 cellules

**Performances :**
- 10 cellules : ~1-2 minutes
- 30 cellules : ~20+ minutes
- 50 cellules : plusieurs heures

**Quand utiliser :**
- Petites simulations (n≤30)
- Besoin de trajectoires optimales
- Pas de GPU disponible

Voir [OXYGEN_SIMULATION_README.md](OXYGEN_SIMULATION_README.md) pour plus de détails.


### 7. Contributeur 

### 8. Référence 




---
