# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from math import sqrt,pi,acos,cos
import matplotlib.pyplot as plt

### Constantes utiles :
phi1=84 ; phi2=204 ; phi3=324
h1=1 #millimètre
b=150 #millimètre
beta=30 # degrés
k=16.6 #rad-1
m=pi/180 #Pour la conversion degrés -> radians


### Sélection de l'image :
# Chargement l'image et conversion en niveau de gris
img = cv.imread("D:\IMT_A\projet_Codevsi\codev 2023-22\pollux_3s.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

### Fonctions intermédiaires pour les calculs :
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


### Modification de l'image et détection des contours :

#On filtre l'image par une filtre médian car il permet de conserver les bords et donc de mieux les détecter que
#si l'on utilisait un filtre moyenneur (type filtre gaussian). Le filtre médian permet de définir la valeur 
#pixel en prennant la valeur médianne de tous ses pixels voisins.
#De plus, cela permet d'éliminer le bruit de poivre et sel sur une image (ce qui apparait souvent sur des
#images astronomiques non prétraité comme dans notre cas).
img_median = cv.medianBlur(gray, 5)

# Seuillage de l'image
Seuil = 20
ret, image_seuillee = cv.threshold(img_median, Seuil, 255, cv.THRESH_BINARY)

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
# On fait un AND entre l'image seuillée et la zone remplie en blanc
th = th | Pdisque

#Enlèvement des bords de l'image
Gdisque = th[bordersize:len(th)-bordersize, bordersize:len(th[0])-bordersize]
Pdisque = Pdisque[bordersize:len(Pdisque)-bordersize, bordersize:len(Pdisque[0])-bordersize]


def detect_contour(image):
    # L'image a déjà été seuillée
    # Appliquer un flou gaussien pour réduire les bruits de l'image
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

### Fonctions pour trouver phi, x0 et les réglages à effectuer :
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

### Fonction pour afficher le résultat :
def afficher():
    # Appeler la fonction detect_contour
    if detect_contour(Gdisque) != None:
        center1, radius1, contours = detect_contour(Gdisque)
        # Dessiner le cercle sur l'image originale
        cv.circle(img, center1, radius1, (0, 0, 255), 2)
        cv.circle(img, center1, 2, (0, 0, 255), 3)
    if detect_contour(Pdisque) != None:
        center2, radius2, contours = detect_contour(Pdisque)
        # Dessiner le cercle sur l'image originale
        cv.circle(img, center2, radius2, (0, 255, 0), 2)
        cv.circle(img, center2, 2, (0, 255, 0), 3)

    # Vecteur de la direction de l'excentricité :
    VectExcentricite = Vecteur(center1,center2)

    #Calcul de l'excentricite :
    diametreGdisque = 2*radius1
    excentrement = Calcul_Excentricite(VectExcentricite,diametreGdisque)

    #Calculs de Phi et x0 :
    phi = PHI(VectExcentricite)


    #Calcul des réglages :
    alpha1 , alpha2 , alpha3 = Reglages(phi,excentrement)

    #Afficher une zone de texte sur l'image
    cv.putText(img, "L'excentrement entre les deux disques est de : " + str(excentrement) , (10, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
    cv.putText(img, "La valeur de phi est de : " + str(phi) + " degres", (10,50), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
    cv.putText(img, "Les reglages a effectuer sont les suivants :" , (10,70), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
    #On donne l'origine du texte qui est le point en bas à gauche du texte
    #(l'origine de l'image étant toujours en haut à gauche), la police de caractères, la taille relative
    #et la couleur.

    #Affichage des instructions pour l'ordre des vis :
    if alpha1 >0:
        cv.putText(img, "Pour la vis 1 : " + str(round(alpha1)) + " degres" , (10,150), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv.putText(img, "Pour la vis 2 : " + str(round(alpha2)) + " degres" , (10,130), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv.putText(img, "Pour la vis 3 : " + str(round(alpha3)) + " degres", (10,110), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
    elif alpha2 > 0:
        cv.putText(img, "Pour la vis 1 : " + str(round(alpha1)) + " degres" , (10,110), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv.putText(img, "Pour la vis 2 : " + str(round(alpha2)) + " degres" , (10,150), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv.putText(img, "Pour la vis 3 : " + str(round(alpha3)) + " degres", (10,130), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
    elif alpha3 >0:
        cv.putText(img, "Pour la vis 1 : " + str(round(alpha1)) + " degres" , (10,110), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv.putText(img, "Pour la vis 2 : " + str(round(alpha2)) + " degres" , (10,130), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv.putText(img, "Pour la vis 3 : " + str(round(alpha3)) + " degres", (10,150), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))

    #Dessine la droite passant par les 2 centres
    cv.line(img, center1, center2, (255,0,0), 2)

    # Afficher l'image avec le cercle
    cv.imshow('Contour detecte', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return