# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 20:07:59 2023

@author: Frédéric
"""

import math
from PIL import Image, ImageDraw, ImageFilter


# Ouvrir l'image et la convertir en niveau de gris
image = Image.open("D:\IMT_A\projet_Codevsi\donut1.png").convert("L")


# Seuillage de l'image
seuil = 1
image_binarisee = image.point(lambda pixel: 255 if pixel >= seuil else 0, mode='1').convert("L")
# On vient appliquer un seuil sur l'image afin de ne prendre en compte que l'anneau, sans les petits anneaux 
# dûs aux phénomènes d'interférences

# Application d'un filtre Gaussien pour lisser l'image
radius = 2
image_Gaussian = image_binarisee.filter(ImageFilter.GaussianBlur(radius=radius)) #FLou gaussian pour éliminer le bruit et
                                                                                 # et les "imperfections"
#image_binarisee.show()
image_Gaussian.show()


# Détection de cercles avec la méthode Hough
# draw = ImageDraw.Draw(image_Gaussian)
# width, height = image.size
# radius = 70
# threshold = 50
# for x in range(radius, width - radius):
#     for y in range(radius, height - radius):
#         pixel = image_Gaussian.getpixel((x, y))
#         if pixel < threshold:
#             count = 0
#             for r in range(radius - 2, radius + 2):
#                 for t in range(0, 360, 5):
#                     a = x - int(r * math.cos(math.radians(t)))
#                     b = y - int(r * math.sin(math.radians(t)))
#                     if 0 <= a < image.width and 0 <= b < image.height:
#                         if image_Gaussian.getpixel((a, b)) < threshold:
#                             count += 1
#             if count > 80:
#                 draw.ellipse((x-radius, y-radius, x+radius, y+radius), 205)

# #Afficher l'image avec les cercles détectés
# image_Gaussian.show()

# Ce code utilise la méthode de transformée de Hough pour la détection de cercles. 
# Il parcourt chaque pixel de l'image et vérifie s'il est en dessous d'un certain seuil de luminosité. 
# S'il l'est, le code vérifie si le nombre de pixels voisins également en dessous de ce seuil 
# correspond à un cercle.
# Le code trace ensuite un cercle autour des pixels détectés avec la méthode draw.ellipse().
