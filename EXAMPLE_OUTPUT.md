# Exemple de Sortie de Simulation

Voici un exemple de sortie de la simulation avec n=10 cellules :

```
⚠ CUDA not functional, using CPU

============================================================
Morphogenesis Simulation with Oxygen-Based Control
============================================================
Configuration:
  • Number of cells: 10
  • State dimension: 21
  • Control dimension: 20
  • Oxygen zones: 3
  • GPU acceleration: Disabled
============================================================

Setting up optimal control problem...
Generating optimal control model...
Solving optimal control problem...
(This may take several minutes for 10 cells)
✓ Solution found! Optimal time: 2.874
Solution extracted: 251 time steps

============================================================
Cell Differentiation Analysis
============================================================

Time t = 0.0:
  • Blood vessels: 0 cells
  • Fibroblasts: 2 cells
  • Base cells: 8 cells

Time t = 1.425:
  • Blood vessels: 6 cells
  • Fibroblasts: 0 cells
  • Base cells: 4 cells

Time t = 2.874:
  • Blood vessels: 0 cells
  • Fibroblasts: 6 cells
  • Base cells: 4 cells
============================================================

Creating visualization...
✓ Static plot saved: morphogenesis_oxygen_n10.png

Creating GIF animation...
✓ GIF animation saved: morphogenesis_oxygen_n10.gif

============================================================
Simulation Complete!
============================================================
Results:
  • Optimal time: 2.874
  • Total cost: 2.874
  • Number of time steps: 251
  • Files generated:
    - morphogenesis_oxygen_n10.png (67 KB)
    - morphogenesis_oxygen_n10.gif (866 KB)
============================================================
```

## Interprétation des Résultats

### Différenciation Cellulaire

L'analyse montre comment les cellules changent de type au cours du temps en fonction de leur position par rapport aux zones d'oxygène :

1. **t = 0.0** (Début) :
   - Les cellules sont en ligne, certaines dans des zones pauvres en oxygène
   - 2 fibroblastes se forment dans les zones à faible oxygène
   - 8 cellules de base dans les zones intermédiaires

2. **t = 1.425** (Mi-parcours) :
   - Les cellules se déplacent vers leur position cible (cercle)
   - 6 vaisseaux sanguins se forment lorsque les cellules traversent des zones riches en oxygène
   - 4 cellules de base maintiennent la stabilité

3. **t = 2.874** (Fin) :
   - Les cellules atteignent la configuration circulaire cible
   - 6 fibroblastes dans les zones à faible oxygène de la périphérie
   - 4 cellules de base dans les zones intermédiaires

### Fichiers Générés

- **PNG** : Image statique montrant la configuration finale
  - Trajectoires des cellules colorées par type
  - Zones d'oxygène en jaune
  - Positions de départ (cercles) et d'arrivée (étoiles)

- **GIF** : Animation complète de la simulation
  - 251 images montrant l'évolution dans le temps
  - Visualisation dynamique de la différenciation cellulaire
  - Taille : ~866 KB pour 10 cellules

### Performance

- **Temps de calcul** : ~1-2 minutes pour 10 cellules sur CPU
- **Temps optimal trouvé** : 2.874 unités de temps
- **Coût total** : 2.874 (minimisé par l'algorithme d'optimisation)

## Comparaison avec les Simulations Précédentes

Les simulations précédentes (R6, R21) utilisaient :
- Un objectif énergétique simple pour chaque cellule
- Pas de différenciation cellulaire
- Moins de réalisme biologique

La nouvelle simulation offre :
- ✅ Contrôle basé sur l'oxygène (plus réaliste)
- ✅ Différenciation cellulaire dynamique
- ✅ Zones environnementales influençant le comportement
- ✅ Visualisation GIF animée
- ✅ Support GPU pour grandes échelles
