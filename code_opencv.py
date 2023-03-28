# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:04:37 2023

@author: Frédéric
"""

import cv2 as cv
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt

def find_circle(img): # l'image doit déjà avoir été filtré, il doit y avoir directement le disque et non le donut
    # blur
    blur = cv.GaussianBlur(img, (3,3), 0)
    # threshold
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # apply morphology open with a circular shaped kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    binary = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # find contour and draw on input (for comparison with circle)
    cnts = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = cnts[0]
    result = img.copy()
    cv.drawContours(result, [c], -1, (0, 255, 0), 1)

    # find radius and center of equivalent circle from binary image and draw circle
    # see https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    # Note: this should be the same as getting the centroid and area=cv2.CC_STAT_AREA from cv2.connectedComponentsWithStats and computing radius = 0.5*sqrt(4*area/pi) or approximately from the area of the contour and computed centroid via image moments.
    regions = measure.regionprops(binary)
    circle = regions[0]
    yc, xc = circle.centroid
    radius = circle.equivalent_diameter / 2.0
    print("radius =",radius, "  center =",xc,",",yc)
    xx = int(round(xc))
    yy = int(round(yc))
    rr = int(round(radius))
    return ((xx,yy),rr)
    
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
im_floodfill_inv = cv.bitwise_not(im_floodfill) # C'est pourquoi nous inversons l'image afin d'avoir la zone à remplir en blanc
th = th | im_floodfill_inv

# Enlèvement des bords de l'image
Gdisque = th[bordersize:len(th)-bordersize, bordersize:len(th[0])-bordersize]

# Redimensionnement de l'image originale pour qu'elle corresponde à la taille du masque
img_resized = cv.resize(image_seuillee, (Gdisque.shape[1], Gdisque.shape[0]))

# Application de la fonction bitwise_and()
resultat = cv.bitwise_and(img_resized, img_resized, mask=Gdisque)




center, radius = find_circle(Gdisque)
cv.circle(Gdisque, center, radius, (255, 255, 0), 3)

# write result to disk
cv.imwrite("dark_circle_fit.png", Gdisque)



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

#hist= cv2.calcHist([img],[i],[256],[0,256])
#