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
    ```julia
    # Exécuter le script principal de simulation
    include("scripts/run_simulation.jl")
    ```

### 6. Structure du Dépôt


### 7. Contributeur 

### 8. Référence 




---
