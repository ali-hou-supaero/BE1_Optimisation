#######################################################################################
# Ce fichier contient la d�finition des algorithmes d'optimisation                    #
#######################################################################################



#######################################################################################
#                                   PARAMETRES EN ENTREE                              #
#######################################################################################

          #   une_f                 une fonction dont on cherche un minimum    

          #   des_c                 les contraintes des probleme               

          #   un_x0                 un incontournable point initial  
          #  un_lambda0             une valeur intiale pour les multiplicateurs de Lagrange.
           

          #   un_gf                 une fonction qui code le gradient de une_f 


          #  jac_des_c              une fonction qui code la jacobienne de des_c 

          #   un_hf                 une fonction qui code le Hessien de  une_f
          #   h_des_c               une fonction qui code les Hessiens de des_c 

          #   un_nit_max            nombre maximum d'iterations autorisees     
          #                         risees                                     
          #   une_tol_x             seuil de stationnarite des x_k             
          #   une_tol_f             seuil de stationnarite des f_k             
          #   une_tol_g             seuil validant EULER    en x_k             
          #                          le sous-probleme:
          #                          (1 exact, 2 pas de Cauchy, 3 CG tronqué)   
          #   une_tol_h             pas de discretisation des derivees 

#######################################################################################
#                                   PARAMETRES EN SORTIE                              #
#######################################################################################           

          #   x_opt                 la solution proposee par trust_region      
          #   f_opt                 une_f (x_opt, varargin{:})                 
          #   g_opt                 un_gf (x_opt, varargin{:})                 
          #   fin                   la cause de l'arret de l'algorithme        
          #   nit                   le nombre iterations de trust_region       
          #   f_count               le nombre d'evaluations de une_f           
          #   g_count               le nombre d'evaluations de un_gf

#  Responsable: E.Flayac (emilien.flayac@isae.fr) -- 2023/2024
#  (C) Institut Superieur de l'Aeronautique et de l'Espace (ISAE-Supaero)
########################################################################################



global f_count, g_count, h_count, c_count, jc_count, hc_count

f_count = 0
g_count = 0
h_count = 0
c_count = 0
jc_count = 0
hc_count = 0






import numpy as np

def algo_SQP(une_f, des_c, un_x0, un_lambda0, un_gf, jac_des_c, un_hf, h_des_c, un_nit_max, une_tol_x, une_tol_f, une_tol_g):

    #tempx et templambda sont des vecteurs colonne
    tempx = np.array(un_x0).reshape(-1, 1)
    templambda = np.array(un_lambda0).reshape(-1, 1)
    
    
    # tempx=np.array(un_x0)
    # templambda = np.array(un_lambda0)   
    n = len(tempx)
    p = len(templambda)
    
    
    #Evaluation du Lagrangien
    fdex = une_f(tempx) # évaluation de la fonction objectif
    cdex = des_c(tempx) # évaluation des contraintes

    #Evaluation du gradient du Lagrangien
    gfdex = un_gf(tempx) # évaluation du gradient de la fonction objectif
    jcdex = jac_des_c(tempx).reshape(1,-1) # évaluation de la jacobienne des contraintes
    
    #Evaluation de la hessienne du Lagrangien
    hfdex = un_hf(tempx) # hessienne de la fonction objectif
    hcdex = h_des_c(tempx) # hessienne des contraintes

    k = 0
    fin = 0
    while fin == 0 and k < un_nit_max:
        
        #A COMPLETER
        # Mise à jour des variables
        # 
        # Construction de la matrice KKT du système
        M = np.block([
            [hfdex + templambda.T * hcdex, jcdex.T],
            [jcdex, np.zeros((p, p))]
        ])

        dklambdak = np.linalg.solve(M, -np.vstack((gfdex, cdex)))
        dk = dklambdak[0:n]
        templambda = dklambdak[n:n+p]

        tempx = tempx + dk
        
        fdex = une_f(tempx)
        cdex = des_c(tempx)
        gfdex = un_gf(tempx)
        jcdex = jac_des_c(tempx).reshape(1,-1)
        hcdex = h_des_c(tempx)
        

        if np.linalg.norm(dk) < une_tol_x:
            fin = 1

        if np.linalg.norm(gfdex + templambda.T @ jcdex) < une_tol_g:
            fin = 2

        if np.linalg.norm(une_f(dk)) < une_tol_f:
            fin = 3

        if k == un_nit_max-1:
            fin = 4
        
        k=k+1

    x_opt = tempx
    f_opt = fdex
    nit = k

    return x_opt, f_opt, fin, nit

def algo_SQP_BFGS(une_f, des_c, un_x0, un_lambda0, un_gf, jac_des_c, H0, un_nit_max, une_tol_x, une_tol_f, une_tol_g):


    tempx = np.array(un_x0).reshape(-1, 1)
    templambda = np.array(un_lambda0).reshape(-1, 1)

    n = len(tempx)
    p = len(templambda)
    
    #Evaluation du Lagrangien
    fdex = une_f(tempx)
    cdex = des_c(tempx)

    #Evaluation du gradient du Lagrangien
    gfdex = un_gf(tempx)
    jcdex = jac_des_c(tempx).reshape(1,-1)

    #Approximation de la hessienne du Lagrangien
    
    H=H0

    k = 0
    fin = 0
    while fin == 0 and k < un_nit_max:
       
        #A COMPLETER
        if k > 0:
            sk_1 = tempx - tempx_old
            yk_1 = (gfdex + jcdex.T @ templambda) - (gfdex_old + jcdex_old.T @ templambda_old)

            if (yk_1.T @ sk_1) > 0:
                H = H + (yk_1 @ yk_1.T)/(yk_1.T @ sk_1) - (H @ sk_1 @ sk_1.T @ H)/(sk_1.T @ H @ sk_1)

        # Résolution du système KKT
        M = np.block([[H, jcdex.T], [jcdex, np.zeros((p,p))]])
        dklambdak = np.linalg.solve(M, -np.vstack((gfdex, cdex)))
        dk = dklambdak[0:n]
        
        # Sauvegarde des anciennes valeurs
        tempx_old, templambda_old = tempx.copy(), templambda.copy()
        gfdex_old, jcdex_old = gfdex.copy(), jcdex.copy()

        # Mise à jour des variables
        tempx = tempx + dk
        templambda = dklambdak[n:n+p]
        fdex = une_f(tempx)
        cdex = des_c(tempx)
        gfdex = un_gf(tempx)
        jcdex = jac_des_c(tempx)


        if np.linalg.norm(dk) < une_tol_x:
            fin = 1

        if np.linalg.norm(gfdex + templambda.T @ jcdex) < une_tol_g:
            fin = 2

        if np.linalg.norm(une_f(dk)) < une_tol_f:
            fin = 3

        if k == un_nit_max-1:
            fin = 4
       
        k=k+1      
            
    x_opt = tempx
    f_opt = fdex
    nit = k

    return x_opt, f_opt, fin, nit




def algo_SQP_sans_derivee(une_f, des_c, un_x0, un_lambda0, H0, une_tol_h, un_nit_max, une_tol_x, une_tol_f, une_tol_g):

    
    
    ######## ATTENTION ###################
    ### Les deux suites de commande suivantes ne donnent pas le meme resultat:
        
    #####cas 1######                          
    # a=np.array([1,2])
    # old_a=a
    # a[0]=3
    # print(a)
    # print(old_a)

    #####cas 2######
    # a=np.array([1,2])
    # old_a=a
    # a=np.array([3,2])
    # print(a)
    # print(old_a)

    ### Afin d'obtenir le meme resultat que le cas 2 en utilisant 
    ### une affectation comme dans le cas 1, il faut utiliser la methode copy()
    
    #####cas 3######                          
    # a=np.array([1,2])
    # old_a=a.copy()
    # a[0]=3
    # print(a)
    # print(old_a)
    



    tempx = np.array(un_x0).reshape(-1, 1)
    templambda = np.array(un_lambda0).reshape(-1, 1)

    n = len(tempx)
    p = len(templambda)
    
    #Evaluation du Lagrangien
    fdex = une_f(tempx)
    cdex = des_c(tempx)

    #Approximation du gradient du Lagrangien
    

    
    #A COMPLETER
    def un_gf(tempx):
        gf = np.zeros((n, 1))
        for i in range(n):
            e_i = np.zeros((n, 1))
            e_i[i] = une_tol_h
            gf[i] = (une_f(tempx + e_i) - une_f(tempx - e_i)) / (2 * une_tol_h)
        return gf
    
    def jac_des_c(tempx):
        jc = np.zeros((p, n))
        for i in range(p):
            for j in range(n):
                e_j = np.zeros((n, 1))
                e_j[j] = une_tol_h
                c_plus = np.array(des_c(tempx + e_j)).reshape(-1)
                c_minus = np.array(des_c(tempx - e_j)).reshape(-1)
                jc[i, j] = (c_plus[i] - c_minus[i]) / (2 * une_tol_h)
        return jc
    
    gfdex = un_gf(tempx)
    jcdex = jac_des_c(tempx).reshape(1,-1)
    

    #Approximation de la hessienne du Lagrangien
    
    H=H0

    k = 0
    fin = 0
    while fin == 0 and k < un_nit_max:
        
        #A COMPLETER
     
        if k > 0:
            sk_1 = tempx - tempx_old
            yk_1 = (gfdex + jcdex.T @ templambda) - (gfdex_old + jcdex_old.T @ templambda_old)

            if (yk_1.T @ sk_1) > 0:
                H = H + (yk_1 @ yk_1.T)/(yk_1.T @ sk_1) - (H @ sk_1 @ sk_1.T @ H)/(sk_1.T @ H @ sk_1)

        # Résolution du système KKT
        M = np.block([[H, jcdex.T], [jcdex, np.zeros((p,p))]])
        dklambdak = np.linalg.solve(M, -np.vstack((gfdex, cdex)))
        dk = dklambdak[0:n]
        
        # Sauvegarde des anciennes valeurs
        tempx_old, templambda_old = tempx.copy(), templambda.copy()
        gfdex_old, jcdex_old = gfdex.copy(), jcdex.copy()

        # Mise à jour des variables
        tempx = tempx + dk
        templambda = dklambdak[n:n+p]
        fdex = une_f(tempx)
        cdex = des_c(tempx)
        gfdex = un_gf(tempx)
        jcdex = jac_des_c(tempx)


        if np.linalg.norm(dk) < une_tol_x:
            fin = 1

        if np.linalg.norm(gfdex + templambda.T @ jcdex) < une_tol_g:
            fin = 2

        if np.linalg.norm(une_f(dk)) < une_tol_f:
            fin = 3

        if k == un_nit_max-1:
            fin = 4
           
        k=k+1   
            
    x_opt = tempx
    f_opt = fdex
    nit = k

    return x_opt, f_opt, fin, nit
