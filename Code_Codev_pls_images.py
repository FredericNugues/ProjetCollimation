from astropy.io import fits
import cv2 as cv
import numpy as np
from math import sqrt,pi,acos,cos


################### Constantes utiles (propres au télescope de la pointe du diable): ###################
phi1=84 ; phi2=204 ; phi3=324
h1=1 #millimètre
b=150 #millimètre
beta=30 # degrés
k=16.6 #rad-1
m=pi/180 #Pour la conversion degrés -> radians


### ------------------- A MODIFIER : -------------------- ###
# Ce chemin est à modifier en fonction du chemin d'accès où se trouvent vos images.
chemin = "C:/Users/thede/Programmation/Projet_Codevsi/sigma_gem_barlow_10s/sigma_gem_barlow_10s-"


### ------------------- Fonctions intermédiaires pour les calculs :--------------------
    
def Vecteur(C1,C2):
    "C1 et C2 sont les coordonnées respectives des centres du grand cercle est du petit cercle"
    "Renvoie les coordonnées du vecteur qui va de C2 à C1"
    (x1,y1) = C1
    (x2,y2) = C2
    X = x1-x2
    Y = y2-y1 #Car le repère n'est pas dans le bon sens : l'origine est en haut à gauche
    V = np.array([[X],[Y]])
    return V

def Norme(vecteur):
    "Renvoie la norme d'un vecteur"
    return sqrt(vecteur[0]**2 + vecteur[1]**2)

def AngleVecteurs(u,v):
    "Renvoie l'angle en degrés"
    N_u = Norme(u)
    N_v = Norme(v)
    pdt_scalaire = u[0]*v[0] + u[1]*v[1]
    a = acos(pdt_scalaire/(N_u*N_v))
    return a/m #On renvoie l'angle en degrés

def Calcul_Excentricite(vecteur,diametre):
    "Le diametre est celui du cercle extérieur"
    "Renvoie l'excentricité des deux cercles"
    norme = Norme(vecteur)
    exc = 2*norme/diametre
    return exc


### ------------------ Modification de l'image et détection des contours :--------------------

def detect_contour(image):
    
    "Fonction permettant de détecter les contours et renvoie le centre, le rayon et le cercle" 
    "qui approxime au mieux l'un des disques rentré en argument de la fonction"
    
    # L'image a déjà été seuillée
    # Application d'un flou gaussien pour réduire les bruits de l'image
    # Ce flou gaussien n'est pas forcément nécessaire car l'image a déjà reçu un filtre médian 
    # permettant d'éliminer les bruits dont le bruit de poivre et sel .
    blur = cv.GaussianBlur(image, (5, 5), 0)
    # Trouver les contours dans l'image binaire
    contours, hierarchy = cv.findContours(blur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Trouver le contour le plus grand
    if not contours:
        print('pas de cercle')
        return None
    max_contour = max(contours, key=cv.contourArea)

    # Approximer le contour par un cercle en utilisant la méthode des moindres carrés
    (x,y), radius = cv.minEnclosingCircle(max_contour)
    center, radius = (int(x), int(y)), int(radius)
    return center, radius, contours


### Fonctions pour trouver l'excentricité x0, son angle associé phi et les réglages à effectuer :
def PHI(vecteur):
    "La variable est le vecteur reliant les deux centres"
    "La fonction renvoie l'angle entre la verticale du centre du 1er cercle et le vecteur"
    x,y = vecteur[0] , vecteur[1]
    S = (0,1)
    if x >= 0:
        phi = AngleVecteurs(vecteur,S)
        return phi
    else :
        angle = AngleVecteurs(vecteur,S)
        phi = 360 - angle
        return phi

def X0(excentricite):
    """Donne x0 à partir de l'excentricité"""
    theta=excentricite/k
    x0=(2*pi*b*theta)/(h1*(cos(m*beta))**2)
    return x0

def Reglages(phi,excentricite):
    x0 = X0(excentricite)

    a1 = 1/m*x0*cos(m*(phi-phi1))
    a2 = 1/m*x0*cos(m*(phi-phi2))
    a3 = 1/m*x0*cos(m*(phi-phi3))

    return a1 , a2 , a3

def calcul_reglages(image,Gdisque,Pdisque):
    ''' Fonction qui calcule les réglages à faire pour une image '''
    # Appeler la fonction detect_contour
    if detect_contour(Gdisque) != None:
        center1, radius1, contours = detect_contour(Gdisque)
        # Dessiner le cercle sur l'image originale
        cv.circle(image, center1, radius1, (0, 0, 255), 2)
        cv.circle(image, center1, 2, (0, 0, 255), 3)
    if detect_contour(Pdisque) != None:
        center2, radius2, contours = detect_contour(Pdisque)
        # Dessiner le cercle sur l'image originale
        cv.circle(image, center2, radius2, (0, 255, 0), 2)
        cv.circle(image, center2, 2, (0, 255, 0), 3)

    # Vecteur de la direction de l'excentricité :
    VectExcentricite = Vecteur(center1,center2)
    
    #Calcul de l'excentricite :
    diametreGdisque = 2*radius1
    excentrement = Calcul_Excentricite(VectExcentricite,diametreGdisque)

    #Calculs de Phi et x0 :
    phi = PHI(VectExcentricite)
    

    #Calcul des réglages :
    alpha1 , alpha2 , alpha3 = Reglages(phi,excentrement)

    return [alpha1,alpha2,alpha3]


### Fonctions pour traiter un lot d'images et renvoyer les réglages à effectuer :
    
def calcul_plusieurs_images(nbImages):
    ''' Fonction qui renvoie une liste avec les réglages à faire pour toutes les images '''
    listeReglages = []
    for i in range(1,nbImages+1):
        nouv_chemin = chemin + str(i) + ".png"
        nouv_chemin_fits = chemin + str(i) + ".fits"
        
        # Chargement l'image et conversion en niveau de gris
        # A utiliser si les images sont au format png:
        #img = cv.imread(chemin + str(i) + ".png")
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        ##Chargement de l'image format fits:
        hdulist = fits.open(nouv_chemin_fits)
        data = hdulist[0].data.astype(np.uint16)

        ##Compression logarithmique afin d'avoir une bonne conversion des données:
        data = np.log10(data + 1)
        data = ((data - np.min(data)) / (np.max(data) - np.min(data))) * 255
        img = np.uint8(data)

        #++++++++++++++++++ Seuillage de l'image+++++++++++++++++
        # Cette variable est extremement importante, c'est elle qui va influer sur la bonne détection 
        # des contours et donc sur la detection des disques. Cette valeur de seuil est à fixer sur 
        # une première image de votre liste d'images de donuts: il s'agit d'utiliser le premier
        # programme en rentrant en paramètre une image de la liste et de jouer sur la valeur du seuil 
        # et voir comment se comporte la détection de cercles. Si aucun cercle n'a été trouvé c'est 
        # que la valeur du seuil est trop grande. Dans ce cas, il suffit de diminuer la valeur de seuil
        # et de continuer jusqu'à ce que les cercles collent au mieux les disques.
        Seuil = 150
        ret, image_seuillee = cv.threshold(img, Seuil, 255, cv.THRESH_BINARY)

        # Ajout de bords à l'image seuillée
        bordersize = 10
        th = cv.copyMakeBorder(image_seuillee, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv.BORDER_CONSTANT, value=[0,0,0])
        # copyMakeBorder permet d’ajouter temporairement des bords vides à l’image
        # afin que la fonction flood fill ne remplisse pas une mauvaise zone


        # Remplissage des contours de l'image seuillée
        im_floodfill = th.copy()
        h, w = th.shape[:2] # On récupère les dimensions hauteur et largeur de l'image
        mask = np.zeros((h+2, w+2), np.uint8)
        cv.floodFill(im_floodfill, mask, (0,0), 255)
        # Le contour à remplir est en noir et le reste en blanc
        # C'est pourquoi nous inversons l'image afin d'avoir la zone à remplir en blanc
        Pdisque = cv.bitwise_not(im_floodfill)
        # On fait un OU INCLUSIF entre l'image seuillée et la zone remplie en blanc
        th = th | Pdisque

        # Enlèvement des bords de l'image
        Gdisque = th[bordersize:len(th)-bordersize, bordersize:len(th[0])-bordersize]

        # Redimensionnement de l'image originale pour qu'elle corresponde à la taille du masque
        img_resized = cv.resize(image_seuillee, (Gdisque.shape[1], Gdisque.shape[0]))

        # Application de la fonction bitwise_and()
        resultat = cv.bitwise_and(img_resized, img_resized, mask=Gdisque)

        #Calcul des réglages pour l'image et ajoute à la liste des résultats :
        reglagesImage = calcul_reglages(img,Gdisque,Pdisque)
        listeReglages.append(reglagesImage)

    return listeReglages

### Fonctions qui calcule la moyenne et l'écart-type des réglages à effectuer :
def MoyenneReglages(listeReglages):
    ''' Fonction qui calcule la moyenne des réglages à effectuer à partir des resultats sur plusieurs images '''
    S1,S2,S3 = 0,0,0
    n = len(listeReglages)
    for reglage in listeReglages:
        S1 += reglage[0]
        S2 += reglage[1]
        S3 += reglage[2]
    moy1 = S1/n
    moy2 = S2/n
    moy3 = S3/n
    return [moy1,moy2,moy3]

def EcartTypeReglages(listeReglages,moyenne):
    ''' Fonction qui calcule l'écart-type des réglages à partir de la liste de réglages et de la moyenne associée '''
    [moy1,moy2,moy3] = moyenne
    S1,S2,S3 = 0,0,0
    n = len(listeReglages)
    for x in listeReglages:
        S1 += (x[0]-moy1)**2
        S2 += (x[1]-moy2)**2
        S3 += (x[2]-moy3)**2
    ET1 = sqrt(S1/n)
    ET2 = sqrt(S2/n)
    ET3 = sqrt(S3/n)
    return [ET1,ET2,ET3]

### Fonction pour afficher les réglages en prenant en compte une série d'images :
def afficherReglages(nbImages):
    # Calcul de tous les réglages :
    listeReglages = calcul_plusieurs_images(nbImages)
    #Calcul de la moyenne :
    moy = MoyenneReglages(listeReglages)

    #Calcul de l'ecart-type :
    ecart_type = EcartTypeReglages(listeReglages,moy)

    #Affichage des résultats :
    print("La vis 1 doit tourner de " + str(moy[0]) + " +- " + str(ecart_type[0]) + " degrés")
    print("La vis 2 doit tourner de " + str(moy[1]) + " +- " + str(ecart_type[1]) + " degrés")
    print("La vis 3 doit tourner de " + str(moy[2]) + " +- " + str(ecart_type[2]) + " degrés")
