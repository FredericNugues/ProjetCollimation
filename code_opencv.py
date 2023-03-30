# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:04:37 2023

@author: Frédéric
"""

import cv2 as cv
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt

def detect_contour(image):
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
    (x, y), radius = cv.minEnclosingCircle(max_contour)
    center, radius = (int(x),int(y)), int(radius)
    return center, radius

    
# Charger l'image et la convertir en niveau de gris
img = cv.imread("D:\IMT_A\projet_Codevsi\donut1.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Seuillage de l'image
Seuil = 1
ret, image_seuillee = cv.threshold(gray, Seuil, 255, cv.THRESH_BINARY)

# Ajout de bords à l'image seuillée
bordersize = 10
th = cv.copyMakeBorder(image_seuillee, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv.BORDER_CONSTANT, value=[0,0,0])
# copyMakeBorder permet d’ajouter temporairement des bords vides à l’image 
# afin que la fonction flood fill ne remplisse pas une mauvaise zone

# Remplissage des contours de l'image seuillée
im_floodfill = th.copy()
h, w = th.shape[:2] # On récupère les dimensions hauteur et largeur de l'image
mask = np.zeros((h+2, w+2), np.uint8)
cv.floodFill(im_floodfill, mask, (0,0), 255) # Le contour à remplir est en noir et le reste en blanc
Pdisque = cv.bitwise_not(im_floodfill) # C'est pourquoi nous inversons l'image afin d'avoir la zone à remplir en blanc
th = th | Pdisque

# Enlèvement des bords de l'image
Gdisque = th[bordersize:len(th)-bordersize, bordersize:len(th[0])-bordersize]

# Redimensionnement de l'image originale pour qu'elle corresponde à la taille du masque
img_resized = cv.resize(image_seuillee, (Gdisque.shape[1], Gdisque.shape[0]))

# Application de la fonction bitwise_and()
resultat = cv.bitwise_and(img_resized, img_resized, mask=Gdisque)



# Appeler la fonction detect_contour
if detect_contour(Gdisque) != None:
    center1, radius1 = detect_contour(Gdisque)
    
    # Dessiner le cercle sur l'image originale
    cv.circle(img, center1, radius1, (0, 0, 255), 2)
    cv.circle(img, center1, 2, (0, 0, 255), 3)
if detect_contour(Pdisque) != None: 
    center2, radius2 = detect_contour(Pdisque)
    
    # Dessiner le cercle sur l'image originale
    cv.circle(img, center2, radius2, (0, 255, 0), 2)
    cv.circle(img, center2, 2, (0, 255, 0), 3)


# Afficher l'image avec le cercle
cv.imshow('Contour detecte', img)
cv.waitKey(0)
cv.destroyAllWindows()

# write result to disk
#cv.imwrite("dark_circle_fit.png", Gdisque)


#cv.imshow('Gdisque', Gdisque)
#cv.waitKey(0)
#cv.destroyAllWindows()

# Enregistrement des images résultantes
#cv.imwrite("im_floodfill.png", im_floodfill) # im_floodfill_inv représente le disque intérieur 
#cv.imwrite("th.png", Gdisque) # th représente le cercle de diamètre le plus grand
#cv.imwrite("resultat.png", resultat)



#cv.imshow('image',gray)# affiche l'image dans une fenêtre intitulé 'image'
#cv2.imwrite('fleur.png',img) #sauvegarde l'image en niveau de gris

#cv.waitKey(0)
#cv.destroyAllWindows()

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #permet de passer de rgb à niveau de gris

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #permet de passer de rgb à HSV

#ret,th=cv2.threshold(img, seuil,couleur, option) 
#Elle prend en paramètre, img : l’image à traiter , seuil : la valeur du seuil , 
#couleur : la couleur que l’on souhaite attribuer à la zone objet et elle retourne 
#ret : la valeur du seuil, et th : l’image binaire résultat du seuillage. Afin d’illustrer cela, dans le code suivant, 
#j’ai seuillé l’image fleur.png préalablement passée en niveau de gris avec 150 comme seuil. 
