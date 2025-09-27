
##############################################################################
#  Ce fichier permet de choisir l'algorithme à éxecuter et le problème à résoudre
# en utilisant la fonction BE1_main à laquelle il faut passer les arguments "algo" et "prob" 
# selon la corespondance suivante : 
#
#
#    algo: 1, 2, ou 3 pour choisir l'algorithme.
#       algo=1 SQP
#       algo=2 SQP avec BFGS
#       algo=3 SQP avec BFGS et différences finies
#    prob: 1, 2 ou 3.
#
# Responsable: E.Flayac (emilien.flayac@isae.fr) -- 
# (C) Institut Supérieur de l'Aéronautique et de l'Espace (ISAE-SUPAERO)
############################################################################






import time

import numpy as np

from BE1_noyau import BE1_main









#**************************
# GLOBALES EN MISE A JOUR *
#**************************


global f_count                  
global g_count                   
global h_count
                    
f_count = 0
g_count = 0
h_count = 0

### Exemples d'utilisation
BE1_main(3,3)



