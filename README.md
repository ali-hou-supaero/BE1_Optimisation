# Optimisation numérique sous contraintes

Bureau d’études réalisé à l’ISAE-SUPAERO autour de la programmation quadratique séquentielle (SQP), en Python avec NumPy.

## Méthodes étudiées

- SQP avec dérivées analytiques.
- SQP avec approximation BFGS de la Hessienne.
- SQP avec BFGS et dérivées estimées par différences finies centrées.

Le noyau propose trois problèmes en deux dimensions : objectif quadratique avec contrainte linéaire, objectif quadratique avec contrainte non linéaire et fonction de Rosenbrock avec contrainte linéaire. Chaque expérience utilise quatre points de départ.

## Organisation

| Fichier | Rôle |
| --- | --- |
| `BE1_algo.py` | Algorithmes d’optimisation |
| `BE1_noyau.py` | Problèmes, dérivées, orchestration et affichage |
| `BE1_main.py` | Sélection de l’expérience |
| `BE1_sujet.pdf` | Énoncé du bureau d’études |

## Utilisation

Avec Python 3 et NumPy, depuis la racine du dépôt :

```sh
python -m pip install numpy
python BE1_main.py
```

Le fichier de lancement appelle actuellement `BE1_main(3, 3)`. Le premier argument sélectionne la méthode (1 : SQP, 2 : BFGS, 3 : différences finies) ; le second sélectionne le problème (1 à 3).

La sortie contient le critère d’arrêt, les nombres d’évaluations, le nombre d’itérations et la valeur de l’objectif. Ce code pédagogique n’est pas un solveur validé pour un usage de production ; aucune garantie de convergence n’est donnée pour toutes les combinaisons.

## Contexte et crédits

Travail effectué à partir du support pédagogique ISAE-SUPAERO attribué à E. Flayac dans les fichiers source. Les mentions d’origine sont conservées. Les instructions ci-dessus sont issues de la lecture du code ; elles n’ont pas été validées par une exécution dans cette passe documentaire.
