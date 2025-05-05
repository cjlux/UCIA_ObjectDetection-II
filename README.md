# Projet UCIA-II

Cette étude vise à entraîner un réseau de neurone de la famille YOLO pour détecter des petits objets 3D placés sur plateau montrant une piste imprimée. Le réseau YOLO est exploité sur une carte RPi4 munie d'une caméra grand angle, placée sur le robot mobile Thymio : les informations sur les objets détectés sont envoyées au robot Thymio qui les traite pour adapter son mouvement selon des choix algoritmiques variés.

L'étude alimente le projet UCIA (Usages et Consciences des Intelligences Artificielles) coordonné par l’association "la ligue de l’enseignement" depuis janvier 2023. Dans le cadre de ce projet, un kit pédagogique doit être créé incluant notamment un robot IA Open Source et Open Hardware dont l’utilisation doit permettre d’encourager un regard critique sur l’Intelligence Artificielle.

Une première étude UCIA-I a visé l'entraînement des réseaux __yolov8n__, __yolov8s__, __yolo11n__ et __yolo11s__ à détecter un nombre limité de petits objets 3D (balle, cube, étoile) placés aur une surface claire. Elle a permis de dégager plusieurs conclusions:
- choix du réseau __yolo8n__ donnant les meilleurs temps d'inférence pour uen précision détection compareble,
- choix des hyper-paramètres d'entraînement `batch_size` et `epochs` conduisant au modèle entraîné le plus performant.

L'étude UCIA-II permet d'aller plus loin :
- 7 objets 3D au lieu des 3 de l'étude précédente : _balle_, _cube_, _étoile_, _maison_, _hexagone_, _cylindre_, _triangle_,
- reconaissnce des objets _virage_gauche_, _virage_droi_, _piste_droite_  formé par la piste noire,
- détection de symboles imprimés : _passage piéton_, _panneau STOP_, _spirale_, _cocarde_...

Le but de l'étude UCIA-II est d'entraîner le réseau __yolov8n__ à détecter et reconnaître un plus grand nombre d'objets de façon à pouvoir alimenter une plus grande diversité d'algotithmes de pilotage du robot Thymio.

Le rapport __UCIA-IA-DetectionObj-II_V1.0.pdf__ du dossier __Doc__ donne plus de détails sur le déroulé de l'étude.


