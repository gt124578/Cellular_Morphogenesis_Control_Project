#!/bin/bash
# Script d'aide pour exécuter les simulations de morphogénèse

echo "============================================"
echo "Simulations de Morphogénèse avec Contrôle Basé sur l'Oxygène"
echo "============================================"
echo ""
echo "Options disponibles:"
echo "  1) Test rapide (10 cellules, ~1-2 minutes)"
echo "  2) Simulation standard (50 cellules, ~10-30 minutes)"
echo "  3) Simulation grande échelle (100 cellules, ~1-3 heures, GPU recommandé)"
echo "  4) Quitter"
echo ""
read -p "Choisissez une option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Lancement du test rapide avec 10 cellules..."
        julia --project=. test_morphogenesis_quick.jl
        ;;
    2)
        echo ""
        echo "Lancement de la simulation avec 50 cellules..."
        echo "Cela peut prendre 10-30 minutes..."
        julia --project=. morphogenesis_n50.jl
        ;;
    3)
        echo ""
        echo "Lancement de la simulation avec 100 cellules..."
        echo "Cela peut prendre 1-3 heures (GPU fortement recommandé)..."
        julia --project=. morphogenesis_n100.jl
        ;;
    4)
        echo "Au revoir!"
        exit 0
        ;;
    *)
        echo "Option invalide!"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Simulation terminée!"
echo "============================================"
echo ""
echo "Fichiers générés:"
ls -lh morphogenesis_oxygen_n*.png morphogenesis_oxygen_n*.gif 2>/dev/null || echo "Aucun fichier trouvé"
