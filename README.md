Ce Projet consiste à apporter une aide à la collimatinon d'un télescope de type "Schmidt cassegrain", présent à la pointe du diable, bretagne.
A partir d'images astronomiques récupérées depuis le télescope, les images contiennent deux disques excentrés, illustrant le défaut de collimation de l'appareil.
![donut1](https://github.com/user-attachments/assets/3decc470-ac30-4bbc-9625-62024bf41d6b)

Tout d'abord, l'idée était de pouvoir traiter une image astronomique en approximant les deux disques par deux cercles dont la distance entre leur centre est reliée au défaut de collimation.
Ensuite, connaissant la valeur de cet excentrement, on peut remonter jusqu'aux réglages des vis de collimations de l'appareil.
Voici le résultat pour une image astronomique dont on observe bien le défaut de collimation:
![Capture résultat 2eme partie](https://github.com/user-attachments/assets/541d530d-8936-4ae1-9725-13ff8738f04b)
Tout cela est contenu dans le script python intitulé "Code_Codev_1_image.py".

Pour finir, le second script python intitulé 'Code_Codev_pls_images.py" permet de faire un traitement de plusieurs images astronomiques et de renvoyer la valeur moyenne des réglages des vis de collimations ainsi que les écart-types respectifs.
