# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

donut1 = Image.open("D:\IMT_A\projet_Codevsi\donut1.png")
donut2 = Image.open("D:\IMT_A\projet_Codevsi\donut2.png")

image1 = io.imread("D:\IMT_A\projet_Codevsi\donut1.png")
#image2 = io.imread("D:\IMT_A\projet_Codevsi\donut2.png")
plt.imshow(image1) # Pour plot l'image 




#donut1.show() #Pour afficher l'image sur une nouvelle fenêtre