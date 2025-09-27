#######################################################################################
# Ce fichier organise l'execution des algorithmes et 
#contient la definition des criteres et des contraintes ainsi que leurs derivees.   
#   
# A NE PAS MODIFIER                                                                                                    
#   
#  Responsable: E.Flayac (emilien.flayac@isae.fr) -- 2023/2024
#  (C) Institut Superieur de l'Aeronautique et de l'Espace (ISAE-Supaero)
########################################################################################






import numpy as np
import time

from BE1_algo import algo_SQP, algo_SQP_BFGS, algo_SQP_sans_derivee



# Fonction c_P1
def c_P1(x):
    global c_count
    c_count += 1
    return x[0,0] + x[1,0] - 1

# Fonction c_P2
def c_P2(x):
    global c_count
    c_count += 1
    return x[0,0]**2 * x[1,0] - 16

# Fonction f_P1
def f_P1(x):
    global f_count
    f_count += 1
    return x[0,0]**2 + x[1,0]**2

# Fonction rosenbrock
def f_rosenbrock(x):
    global f_count
    f_count += 1
    return 100 * (x[1,0] - x[0,0]**2)**2 + (1 - x[0,0])**2

# Fonction g_f_P1
def g_f_P1(x):
    global g_count
    g_count += 1
    return np.array([[2 * x[0,0]], [2 * x[1,0]]])

# Fonction g_rosenbrock
def g_rosenbrock(x):
    global g_count
    g_count += 1
    return np.array([[-400 * (x[1,0] - x[0,0]**2) * x[0,0] - 2 * (1 - x[0,0])], [200 * (x[1,0] - x[0,0]**2)]])

# Fonction h_c_P1
def h_c_P1(x):
    global hc_count
    hc_count += 1
    return np.array([[0, 0], [0, 0]])

# Fonction h_c_P2
def h_c_P2(x):
    global hc_count
    hc_count += 1
    return np.array([[2 * x[0,0] * x[1,0], 2 * x[0,0]], [2 * x[0,0], 0]])

# Fonction h_f_P1
def h_f_P1(x):
    global h_count
    h_count += 1
    return np.array([[2, 0], [0, 2]])

# Fonction h_rosenbrock
def h_rosenbrock(x):
    global h_count
    h_count += 1
    return np.array([[1200 * x[0,0]**2 - 400 * x[1,0] + 2, -400 * x[0,0]], [-400 * x[0,0], 200]])

# Fonction jac_c_P1
def jac_c_P1(x):
    global jc_count
    jc_count += 1
    return np.array([1, 1]).reshape(1,-1)

# Fonction jac_c_P2
def jac_c_P2(x):
    global jc_count
    jc_count += 1
    return np.array([2 * x[0,0] * x[1,0], x[0,0]**2]).reshape(1,-1)















def BE1_main(algo, prob):
    np.seterr(all='ignore')  # Ignorer les avertissements numériques
    
    # Paramètres globaux
    global f_count, g_count, h_count, c_count, jc_count, hc_count
    f_count, g_count, h_count, c_count, jc_count, hc_count = 0, 0, 0, 0, 0, 0

    # En-têtes pour l'affichage
    ligne_tiret = '-------------------------------------------------------'
    ligne_tiret = '|' + ligne_tiret + ligne_tiret + '|'
    
    lentete = '| DEPART|      METHODE      | PROBLEME |  FIN  |   F_COUNT   | ' \
              ' G_COUNT   |  H_COUNT |   NITER    |   F_OPT    |'
    
    les_formats = '|%s| %s |%s |  %3.0f  |   %6.0f    |     %5.0f  |   %5.0f  |' \
                  '    %5.0f   |%10.5g  |'
    
    nom_algo = ['SQP avec derivees', '  SQP avec BFGS  ', ' SQP sans derivee']
    nom_prob = ['    P1   ', '    P2   ', '   P3    ']
    
    
    
    # Initialisaiton des vecteurs
    top_on = [0] * 4
    x_opt = [[0] * 2 for _ in range(4)]
    f_opt = [0] * 4
    conv = [0] * 4
    ite = [0] * 4
    temps = [0] * 4
    x_optimale = [[[0] * 2 for _ in range(4)] for _ in range(6)]
    t_cpu_time = [[0] * 4 for _ in range(6)]
    t_fin = [[0] * 4 for _ in range(6)]
    t_f_count = [[0] * 4 for _ in range(6)]
    t_g_count = [[0] * 4 for _ in range(6)]
    t_h_count = [[0] * 4 for _ in range(6)]
    t_nit = [[0] * 4 for _ in range(6)]
    t_f_opt = [[0] * 4 for _ in range(6)]

    
    
    
    
    
    
    
    

    # Points de départ
    x = np.array([[1, 1], [2, 2], [10, 10], [2, 5]])
    nom_point = [' (1,1) ', ' (2,2) ', '(10,10)', ' (2,5) ']

    # Choix du problème de minimisation (P1, P2 ou P3)
    if prob == 1:
        f_fun, g_fun, h_fun, c_fun, jac_c_fun, h_c_fun = f_P1, g_f_P1, h_f_P1, c_P1, jac_c_P1, h_c_P1
        lambda0 = [0]
        H0 = np.eye(2)
    elif prob == 2:
        f_fun, g_fun, h_fun, c_fun, jac_c_fun, h_c_fun = f_P1, g_f_P1, h_f_P1, c_P2, jac_c_P2, h_c_P2
        lambda0 = [0]
        H0 = np.eye(2)
    elif prob == 3:
        f_fun, g_fun, h_fun, c_fun, jac_c_fun, h_c_fun = f_rosenbrock, g_rosenbrock, h_rosenbrock, c_P1, jac_c_P1, h_c_P1
        lambda0 = [0]
        H0 = np.eye(2)

    # Algorithme SQP
    if algo == 1:
        for i in range(4):  # Pour chaque point de départ
        
            f_count    = 0                                                             
            g_count    = 0                                                             
            h_count    = 0                                                             

            c_count    = 0                                                             
            jc_count   = 0                                                             
            hc_count   = 0   
        
            top_on = np.zeros(4)
            top_on[i] = time.process_time()
            x_opt, f_opt, conv, ite = algo_SQP(f_fun, c_fun, x[i], lambda0, g_fun, jac_c_fun, h_fun, h_c_fun,
                                               100, 1e-5, 1e-5, 1e-5)

            # Garder les sorties pour l'affichage
            temps = time.process_time() - top_on[i]
            x_optimale[:][i] [algo - 1] = x_opt
            t_cpu_time[algo - 1][i] = temps
            t_fin[algo - 1][i] = conv
            t_f_count[algo - 1][i] = f_count
            t_g_count[algo - 1][i] = g_count
            t_h_count[algo - 1][i] = h_count
            t_nit[algo - 1][i] = ite
            t_f_opt[algo - 1][i] = f_opt
            
        f_count    = 0                                                             
        g_count    = 0                                                             
        h_count    = 0                                                             

        c_count    = 0                                                             
        jc_count   = 0                                                             
        hc_count   = 0   
        
        
    # Algorithme SQP avec BFGS
    if algo == 2:
        
        
        
        for i in range(4):  # Pour chaque point de départ
        
            f_count    = 0                                                             
            g_count    = 0                                                             
            h_count    = 0                                                             

            c_count    = 0                                                             
            jc_count   = 0                                                             
            hc_count   = 0   
            
            top_on = np.zeros(4)
            top_on[i] = time.process_time()
            x_opt, f_opt, conv, ite = algo_SQP_BFGS(f_fun, c_fun, x[i], lambda0, g_fun, jac_c_fun, H0,
                                                    100, 1e-5, 1e-5, 1e-5)

            # Garder les sorties pour l'affichage
            temps = time.process_time() - top_on[i]
            x_optimale[:][i][algo - 1] = x_opt
            t_cpu_time[algo - 1][i] = temps
            t_fin[algo - 1][i] = conv
            t_f_count[algo - 1][i] = f_count
            t_g_count[algo - 1][i] = g_count
            t_h_count[algo - 1][i] = h_count
            t_nit[algo - 1][i] = ite
            t_f_opt[algo - 1][i] = f_opt
            
        f_count    = 0                                                             
        g_count    = 0                                                             
        h_count    = 0                                                             

        c_count    = 0                                                             
        jc_count   = 0                                                             
        hc_count   = 0   
    # Algorithme SQP sans dérivée
    if algo == 3:
        for i in range(4):  # Pour chaque point de départ
        
        
            f_count    = 0                                                             
            g_count    = 0                                                             
            h_count    = 0                                                             

            c_count    = 0                                                             
            jc_count   = 0                                                             
            hc_count   = 0   
            
            top_on = np.zeros(4)
            top_on[i] = time.process_time()
            x_opt, f_opt, conv, ite = algo_SQP_sans_derivee(f_fun, c_fun, x[i],lambda0, H0,1e-5, 100,
                                                           1e-5, 1e-5,1e-5)

            # Garder les sorties pour l'affichage
            temps = time.process_time() - top_on[i]
            x_optimale[:][i][algo - 1] = x_opt
            t_cpu_time[algo - 1][i] = temps
            t_fin[algo - 1][i] = conv
            t_f_count[algo - 1][i] = f_count
            t_g_count[algo - 1][i] = g_count
            t_h_count[algo - 1][i] = h_count
            t_nit[algo - 1][i] = ite
            t_f_opt[algo - 1][i] = f_opt
            
        f_count    = 0                                                             
        g_count    = 0                                                             
        h_count    = 0                                                             

        c_count    = 0                                                             
        jc_count   = 0                                                             
        hc_count   = 0       
            

    # Affichage des résultats
    print('\n' + ligne_tiret)
    print(lentete)
    print(ligne_tiret)

    for i in range(4):  # Pour chaque point de départ
        print(les_formats % (nom_point[i], nom_algo[algo - 1], nom_prob[prob - 1], t_fin[algo - 1][i],
                            t_f_count[algo - 1][i], t_g_count[algo - 1][i], t_h_count[algo - 1][i],
                            t_nit[algo - 1][i], t_f_opt[algo - 1][i]))

    print(ligne_tiret)
