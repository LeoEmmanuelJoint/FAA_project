
import os
from builtins import print
import cv2
import shutil
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
import numpy as np

"""
Copie d'image pour completer nos données
"""

def copy_images_from_train(train_folder, extracted_folder, emotion_mapping):
    print("Copie des images depuis le dossier 'train' en cours...")
    # Parcours des dossiers d'émotions extraits des vidéos
    for emotion_folder in emotion_mapping.values():
        # Chemin du dossier d'émotion extrait des vidéos
        extracted_emotion_folder = os.path.join(extracted_folder, emotion_folder)

        # Vérification si le dossier d'émotion existe
        if not os.path.exists(extracted_emotion_folder):
            os.makedirs(extracted_emotion_folder)

    # Vérification si le dossier "neutral" existe
    neutral_folder = os.path.join(extracted_folder, "neutral")
    if not os.path.exists(neutral_folder):
        os.makedirs(neutral_folder)

    # Copie des images du dossier "train" vers les dossiers d'émotions correspondants
    for root, dirs, files in os.walk(train_folder):
        for file in files:
            # Chemin complet de l'image source
            image_path = os.path.join(root, file)

            # Analyse de l'étiquette d'émotion à partir du nom du dossier parent
            emotion_label = os.path.basename(root)

            # Vérification si l'étiquette d'émotion a une correspondance dans le mapping
            if emotion_label in emotion_mapping:
                # Chemin du dossier d'émotion extrait des vidéos
                extracted_emotion_folder = os.path.join(extracted_folder, emotion_mapping[emotion_label])

                # Copie de l'image vers le dossier d'émotion correspondant
                destination_path = os.path.join(extracted_emotion_folder, file)
                shutil.copy(image_path, destination_path)
            else:
                # Copie de l'image dans le dossier "neutral"
                destination_path = os.path.join(neutral_folder, file)
                shutil.copy(image_path, destination_path)

    print("Copie des images depuis le dossier 'train' terminée.")

############################################################################################################

"""
Extraction des données de vidéos du dataset Ryerson Emotion Database
 et ajout des données du dataset emotion detection pour compléter les données
"""

def extract(base_folder, destination_folder):
    # Chemin vers le dossier contenant les vidéos
    videos_folder = base_folder

    # Chemin vers le dossier de sortie pour les images extraites
    output_folder = destination_folder

    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # Chargement du détecteur de visage Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Mapping des étiquettes d'émotions aux dossiers correspondants
    emotion_mapping = {
        'ha': 'happiness',
        'sa': 'sadness',
        'an': 'anger',
        'fe': 'fear',
        'su': 'surprise',
        'di': 'disgust'
    }

    # Vérification si le dossier de sortie est vide
    if not os.listdir(output_folder):
        # Création des dossiers d'émotions dans le dossier de sortie
        for emotion_folder in emotion_mapping.values():
            emotion_output_folder = os.path.join(output_folder, emotion_folder)
            os.makedirs(emotion_output_folder, exist_ok=True)

        # Parcours des vidéos dans le dossier
        for root, dirs, files in os.walk(videos_folder):
            for file in files:
                # Vérification de l'extension de fichier vidéo
                if file.endswith(".mp4") or file.endswith(".avi"):
                    # Chemin complet du fichier vidéo
                    video_path = os.path.join(root, file)

                    # Lecture de la vidéo
                    video_capture = cv2.VideoCapture(video_path)

                    # Variables pour le comptage des images extraites
                    image_count = 0
                    total_images = 0

                    # Analyse de l'étiquette d'émotion à partir du nom du dossier de la vidéo
                    emotion_label = file[:2]
                    emotion_folder = emotion_mapping.get(emotion_label)

                    if emotion_folder is not None:
                        # Calcul de la durée de la vidéo
                        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
                        duration = total_frames / frame_rate

                        # Calcul des indices des frames à extraire
                        start_frame = int(frame_rate * 1.5)
                        end_frame = total_frames - int(frame_rate * 1.5)

                        # Positionnement à la frame de début
                        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                        # Parcours des trames de la vidéo
                        while video_capture.isOpened():
                            # Lecture de la trame
                            ret, frame = video_capture.read()

                            if not ret or video_capture.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                                break

                            # Détection des visages dans la trame
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                            # Parcours des visages détectés
                            for (x, y, w, h) in faces:
                                # Extraction du visage
                                face = gray[y:y+h, x:x+w]

                                # Sauvegarde de l'image du visage extrait en nuances de gris
                                image_name = f"image_{image_count}.jpg"
                                image_path = os.path.join(output_folder, emotion_folder, image_name)
                                
                                # Redimensionner l'image
                                face = cv2.resize(face, (48, 48))
                                
                                cv2.imwrite(image_path, face)

                                # Incrémentation du compteur d'images
                                image_count += 1

                                # Affichage du progrès
                                print(f"Extraction des images - Vidéo : {file} - Images extraites : {image_count}", end="\r")

                                # Limite le nombre d'images extraites pour tester
                                # Supprimez cette condition pour extraire tous les visages de chaque vidéo
                                if image_count >= 100:
                                    break

                                # Incrémentation du compteur d'images total
                                total_images += 1

                        # Affichage des statistiques pour la vidéo traitée
                        print(f"\nExtraction terminée - Vidéo : {file} - Total des images : {total_images} - Images extraites : {image_count}")

                    # Fermeture de la capture vidéo
                    video_capture.release()

        #Copie des données complémentaire pour le dataset

        # Dossier "train" contenant les images à copier
        train_folder = "train"

        # Dossier extrait des vidéos contenant les dossiers d'émotions
        extracted_folder = "extract/data"

        # Mapping des étiquettes d'émotions entre le dossier "train" et les dossiers extraits des vidéos
        emotion_mapping = {
            'happy': 'happiness',
            'sad': 'sadness',
            'angry': 'anger',
            'fearfull': 'fear',
            'surprised': 'surprise',
            'disgusted': 'disgust'
        }

        # Appel de la fonction de copie des images
        copy_images_from_train(train_folder, extracted_folder, emotion_mapping)

        print("Extraction des visages à partir des vidéos terminée.")
    else:
        print("Le dossier de sortie contient déjà des images. Aucune extraction nécessaire.")


########################################################################################################################

"""
décompte des données dans chaque classe d'émotion
"""
def count_images_per_emotion(extracted_folder):
    print(f"Décompte des labels du dossier {extracted_folder}")
    # Parcours des dossiers d'émotions extraits des vidéos
    for root, dirs, files in os.walk(extracted_folder):
        # Vérification si le dossier contient des fichiers
        if files:
            # Récupération du nom du dossier d'émotion
            emotion_folder = os.path.basename(root)

            # Comptage du nombre d'images dans le dossier
            image_count = len(files)

            # Affichage du nombre d'images par dossier d'émotion
            print(f"{emotion_folder}: {image_count} image(s)")
    print("")


########################################################################################################################

"""
Compter les nouvelles classes après avoir appliquer le méthode smote
"""
def count_samples(image_paths, labels):
    # Compter le nombre d'échantillons dans chaque classe
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # Afficher les résultats
    for emotion_label, count in zip(unique_classes, class_counts):
        print(f"Classe {emotion_label}: {count} échantillons")

"""
Préparation des données pour l'entrainement
"""

def process_data(data_folder, emotion_labels):
    # Liste pour stocker les images
    images = []
    # Liste pour stocker les étiquettes correspondantes aux images
    labels = []

    # Nombre maximum d'images à utiliser par classe émotionnelle
    max_images_per_class = 6000 # à retirer plus tard

    # Parcours des dossiers d'émotions
    for emotion_label in emotion_labels:
        emotion_folder = os.path.join(data_folder, emotion_label)
        # Parcours des fichiers d'images dans chaque dossier d'émotion
        image_count = 0
        for image_file in os.listdir(emotion_folder):
            if image_count >= max_images_per_class:
                break
            image_path = os.path.join(emotion_folder, image_file)
            # Charger l'image à partir du chemin
            image = cv2.imread(image_path)
            # Ajouter l'image et l'étiquette aux listes correspondantes
            images.append(image)
            labels.append(emotion_label)
            image_count += 1

    # Encodage des étiquettes d'émotions en valeurs numériques
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Convertir les images en tableaux numpy
    images = np.array(images)

    # Mélange aléatoire des données
    images, labels = shuffle(images, labels, random_state=42)

    # Utilisation de SMOTE pour sur-échantillonner les classes minoritaires
    smote = SMOTE(random_state=42)
    images, labels = smote.fit_resample(images.reshape(len(images), -1), labels)

    # Affichage des statistiques
    print("Nombre total d'images :", len(images))
    print()

    # Appeler la fonction pour compter les échantillons dans chaque classe
    count_samples(labels)

def count_samples(labels):
    # Compter le nombre d'échantillons dans chaque classe
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # Afficher les résultats
    for emotion_label, count in zip(unique_classes, class_counts):
        print(f"Classe {emotion_label}: {count} échantillons")


def count_samples(labels):
    # Compter le nombre d'échantillons dans chaque classe
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # Afficher les résultats
    for emotion_label, count in zip(unique_classes, class_counts):
        print(f"Classe {emotion_label}: {count} échantillons")



#appel de la fonctio pour l'extraction des données
extract("data", "extract/data")


# Chemin du dossier d'entrainement
extracted_folder_train = "extract/data"
# Chemin du dossier de test
extracted_folder_test = "test"

# Appel de la fonction pour compter les images dans le dossier pour l'entrainement
count_images_per_emotion(extracted_folder_train)
# Appel de la fonction pour compter les images dans le dossier pour le test
count_images_per_emotion(extracted_folder_test)

# Appeler la fonction pour traiter les données
data_folder = "extract/data"
emotion_labels = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
process_data(data_folder, emotion_labels)