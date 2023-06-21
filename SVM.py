import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, auc, roc_curve
import os
import joblib
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score


def count_samples(labels):
    # Compter le nombre d'échantillons dans chaque classe
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # Afficher les résultats
    for emotion_label, count in zip(unique_classes, class_counts):
        print(f"Classe {emotion_label}: {count} échantillons")
##################################################################################
def process_data(data_folder, emotion_labels, images_list, labels_list, type):

    # Parcours des dossiers d'émotions
    for emotion_label in emotion_labels:
        emotion_folder = os.path.join(data_folder, emotion_label)
        # Parcours des fichiers d'images dans chaque dossier d'émotion
        image_count = 0
        for image_file in os.listdir(emotion_folder):
            image_path = os.path.join(emotion_folder, image_file)
            # Charger l'image à partir du chemin
            image = cv2.imread(image_path)
            # Ajouter l'image et l'étiquette aux listes correspondantes
            images_list.append(image)
            labels_list.append(emotion_label)
            image_count += 1

    # Encodage des étiquettes d'émotions en valeurs numériques
    label_encoder = LabelEncoder()
    labels_list = label_encoder.fit_transform(labels_list)

    # Convertir les images en tableaux numpy
    images_list = np.array(images_list)

    # Mélange aléatoire des données
    images_list, labels_list = shuffle(images_list, labels_list, random_state=42)

    if(type=="train"):
        # Utilisation de SMOTE pour sur-échantillonner les classes minoritaires
        smote = SMOTE(random_state=42)
        images_list, labels_list = smote.fit_resample(images_list.reshape(len(images_list), -1), labels_list)

    # Affichage des statistiques
    print("Nombre total d'images :", len(images_list))
    print()

    # Appeler la fonction pour compter les échantillons dans chaque classe
    count_samples(labels_list)
##################################################################################


"""
entrainement
"""


# Chemin vers le dossier contenant les images de test
train_data_folder = "D:/UQAC ETE 2023/8INF867_Fondamentaux de l'apprentissage automatique/Projet/FAA_Project/FAA_project/data"  # Mettez le bon chemin vers votre dossier de test
test_data_folder = "D:/UQAC ETE 2023/8INF867_Fondamentaux de l'apprentissage automatique/Projet/FAA_Project/FAA_project/new styve/test"  # Mettre le bon chemin
emotion_labels = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# Charger les données d'entraînement
train_images = []
train_labels = []
process_data(train_data_folder, emotion_labels, train_images, train_labels, "train")

# Charger les données de test
test_images = []
test_labels = []
process_data(test_data_folder, emotion_labels, test_images, test_labels, "test")

# Appliquer PCA sur les données d'entraînement
pca = PCA(n_components=100)  # Définir le nombre de composantes principales souhaité

# Convertir la liste d'images en un tableau numpy
train_images_array = np.array(train_images)

# Aplatir les images d'entraînement
train_images_flat = train_images_array.reshape(len(train_images_array), -1)

# Appliquer PCA sur les images aplaties
train_images_pca = pca.fit_transform(train_images_flat)


# Créer un modèle SVM
svm_model = SVC(probability=True)

# Entraîner le modèle sur les données d'entraînement PCA
svm_model.fit(train_images_pca, train_labels)

# Sauvegarder le modèle entraîné
joblib.dump(svm_model, 'svm_model.pkl')

# Charger le modèle à partir du fichier
loaded_svm_model = joblib.load('svm_model.pkl')

# Convertir la liste d'images de test en un tableau numpy
test_images_array = np.array(test_images)

# Aplatir les images de test
test_images_flat = test_images_array.reshape(len(test_images_array), -1)

# Appliquer PCA sur les images aplaties de test
test_images_pca = pca.transform(test_images_flat)


# Prédire les étiquettes pour les données de test
predictions = loaded_svm_model.predict(test_images_pca)

# Obtenir les probabilités de prédiction pour les données de test
probabilities = loaded_svm_model.predict_proba(test_images_pca)


# Calculer les prédictions pour les données de test
predictions = loaded_svm_model.predict(test_images_pca)

# # Calculer la précision
# precision = precision_score(test_labels, predictions, average='weighted')
#
# # Calculer le rappel
# recall = recall_score(test_labels, predictions, average='weighted')
#
# # Calculer la f-mesure
# f1 = f1_score(test_labels, predictions, average='weighted')

# Afficher les métriques
print()
print("Rapport de classification:")
print(classification_report(test_labels, predictions))

# Calculer la précision du modèle
accuracy = accuracy_score(test_labels, predictions)
print("Exactitude du modèle SVM :", accuracy)
