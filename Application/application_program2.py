##
# @file application_program2.py
# Principal file containing the python program
# @author Léo-Emmanuel JOINT
# @date Juney 2023

# @remark
# In this program, the names of the elements are idetified with prefixes and suffixes
# - the Tkinter control variables are named with the suffix "_gui"
# - the Tkinter.ttk.Combobox elements are identified with the prefix "cbb_"
# - the Tkinter.ttk.Frame elements are identified with the prefix "f_"
# - the Tkinter.ttk.LabelFrame elements are identified with the prefix "lbf_"
# - the Tkinter.ttk.RadioButton elements are identified with the prefix "rbtn_"
# - the Tkinter.ttk.Checkbutton elements are identified with the prefix "chb_"
# - the Tkinter.ttk.Button elements are identified with the prefix "btn_"
# - the Tkinter.ttk.Label and Tkinter.Label elements are identified with the prefix "lbl_"
# - the Tkinter.ttk.Spinbox elements are identified with the prefix "spb_"

# @warning
# The application may take some time to setup

# Program that shows the GUI
import cv2

print("Importing resources, please wait...")
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os
from keras.models import model_from_json
from joblib import load
from sklearn.decomposition import PCA
from skimage.feature import hog
import pickle

print("Done")
print("Setting the interface...")

# Root window
MyWindow = tk.Tk()

# STYLE

# Theme to use in the interface (sets a style for each gui element)
# Not all themes are avaliable depending on the distribution (Windows, Linux or Mac os)
s1 = ttk.Style()
# print("Available themes:", s1.theme_names()) #('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative') on Windows 10
# print("Theme in use:", s1.theme_use()) # vista
s1.theme_use('clam')  # Sets the theme

##
# Font for the species displayed (italic)
titleFont = ('Helvetica', 16, 'bold')
frameRate = 25

# Paths to the models and their results by default

pathDict = {
    'pathModel_CNN': "FERmodel_CNN.json",
    'pathModel_weights_CNN': "FERmodel_CNN.h5",
    'pathRes_CNN': "history_dict_bin.pkl",
    'pathRes_SVM': "../SVM/history_dict_svm1_bin.pkl",
    'pathRes_RFC': "test_performance_metrics_bin.pkl",
    'pathModel_SVM': "../SVM/svm_model.pkl",
    'pathModel_RFC': "emotion_model.pkl",
    'pathHaarCascade': "haarcascade_frontalface_default.xml",
    'pathModel_PCA': "pca_model1.pkl"
}

emotionsDict = {
    0: 'angry',
    1: 'disgusted',
    2: 'afraid',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'}


# CLASSES

class Processing:
    """ Manages all the image processing, loads the models"""

    def __init__(self, dictPaths, emotionsDict):

        self.pathResultsCNN = dictPaths['pathRes_CNN']
        self.pathResultsSVM = dictPaths['pathRes_SVM']
        self.pathResultsRFC = dictPaths['pathRes_RFC']
        self.pathModelResults = ""

        self.pathModelCNN = dictPaths['pathModel_CNN']
        self.pathWeightsCNN = dictPaths['pathModel_weights_CNN']
        self.pathModelSVM = dictPaths['pathModel_SVM']
        self.pathModelRFC = dictPaths['pathModel_RFC']
        self.pathHaarCascade = dictPaths['pathHaarCascade']
        self.pathPCA = dictPaths['pathModel_PCA']

        self.face_detector = cv.CascadeClassifier(self.pathHaarCascade)

        self.dict_emotions = emotionsDict
        self.dict_res = dict()

        self.dict_modelNames = {
            0: "CNN",
            1: "SVM",
            2: "RFC"
        }

        self.FERModel = None
        self.currentModelName = ""

        self.pca = PCA(n_components=60)

    def getEmotion(self, emotionId):
        return self.dict_emotions[emotionId]

    def getModelName(self, modelId):
        return self.dict_modelNames[modelId]

    def load_model(self, idModel):

        modelName = self.dict_modelNames[idModel]
        print("Loading " + modelName + " model ...")

        if idModel == 0:
            self.FERModel = model_from_json(open(self.pathModelCNN, "r").read())
            self.FERModel.load_weights(self.pathWeightsCNN)
            self.pathModelResults = self.pathResultsCNN

        elif idModel == 1:
            self.FERModel = load(self.pathModelSVM)
            self.pathModelResults = self.pathResultsSVM

        elif idModel == 2:
            self.FERModel = load(self.pathModelRFC)
            self.pathModelResults = self.pathResultsRFC

        self.currentModelName = modelName

        print("Model loaded")

    def extract_features(self, image):
        # Appliquer le descripteur HOG
        hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        return hog_features

    def detect_on_image_path(self, pathImage):
        print("Opening file...")
        image = cv.imread(pathImage)
        print("File opened")
        print("Beginning detection on image...")
        return self.detect_on_image(image)

    def get_prediction_nb_RFC(self, face_img):
        # Charger le modèle RandomForestClassifier à partir du fichier
        model = pickle.load(open(self.pathModelRFC, 'rb'))

        try:
            loaded_pca = pickle.load(open(self.pathPCA, 'rb'))
        except FileNotFoundError:
            loaded_pca = None

        if loaded_pca is None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=60)
            pca.fit(face_img)
        else:
            pca = loaded_pca

        # Prétraiter l'image
        resized_image = cv2.resize(face_img, (48, 48))

        # Extraire les caractéristiques (HOG) de l'image
        hog_features = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                           visualize=False)

        # Réduire les caractéristiques de l'image en utilisant les composantes principales de PCA
        features_pca = pca.transform([hog_features])

        # Prédire l'émotion
        emotion_index = model.predict(features_pca)[0]

        emotion = emotionsDict[emotion_index]

        # Retourner la prédiction sous forme d'entier
        return emotion_index



    def get_prediction_nb_CNN(self, face_img):

        input_image = cv.resize(face_img, (48, 48))
        input_image = input_image / 255
        input_image = np.expand_dims(input_image, axis=0)

        # Predict the emotion
        predictions = self.FERModel.predict(input_image)
        best = np.argmax(predictions[0])
        return (best)

    def detect_on_image(self, image):

        image_copy = image.copy()
        image_shape = image_copy.shape
        print("Shape: ", image_shape)

        # Convert the image to gray scale OpenCV
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Detect the faces using haar cascade classifier
        faces_coordinates = self.face_detector.detectMultiScale(gray_img)

        for (x, y, w, h) in faces_coordinates:

            # Crop the face from the image
            cropped_face = gray_img[y:y + h, x:x + w]

            if self.currentModelName == "CNN":
                prediction = self.get_prediction_nb_CNN(cropped_face)
            elif self.currentModelName == "RFC":
                prediction = self.get_prediction_nb_RFC(cropped_face)

            emotion_predicted = self.dict_emotions[prediction]

            print("Detection " + emotion_predicted)

            # Draw rectangle around the face
            cv.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Write the corresponding emotion
            cv.putText(image_copy, emotion_predicted, (int(x), int(y + h)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return (image_copy)

    def get_training_results(self):
        with open(self.pathModelResults, 'rb') as f:
            # f.seek(0)
            self.dict_res = pickle.load(f)
        conf_matrix = self.dict_res["test_cf_matrix"]
        conf_matrix = np.array2string(conf_matrix, precision=2, separator=' ', suppress_small=True)
        accuracy = self.dict_res["test_accuracy"]
        kappa = self.dict_res["test_kappa"]
        report = self.dict_res["test_report"]
        res = "Accuracy: " + str(accuracy) + \
              "\n\nKappa: " + str(kappa) + \
              "\n\nClassification report: \n" + report + "\n\nConfusion matrix: \n" + conf_matrix
        print(res)
        return (res)


class Interface(ttk.Frame):
    """ Class that manages all the interface elements """

    def __init__(self, mainframe, dict_paths, emotionsDict, frameRate):
        ttk.Frame.__init__(self, master=mainframe)
        self.pack(side='top', expand=True, fill='both')
        self.master.title("Face Emotion Recognition test program")
        self.frameDuration = int(1000 / frameRate)
        self.monoFont = ('Courier', 9)

        self.defaultTestResultText = "Please select a model."
        self.loopOnVideoStream = False
        self.imageDisplayed = None
        self.capture = None

        ## VARIABLES

        self.modelId_gui = tk.IntVar(value=2)

        self.process = Processing(dict_paths, emotionsDict=emotionsDict)

        ## USER INTERFACE

        ### Selection part for the model

        self.lbf_modelSelection = ttk.LabelFrame(self, text="Select the model")
        self.lbf_modelSelection.pack(side='top', expand=False, fill='both', anchor='w', padx=3, pady=2)

        self.rbtn_CNN = ttk.Radiobutton(self.lbf_modelSelection, text="CNN", variable=self.modelId_gui, value=0,
                                        cursor="hand2")
        self.rbtn_SVM = ttk.Radiobutton(self.lbf_modelSelection, text="SVM", variable=self.modelId_gui, value=1,
                                        cursor="hand2")
        self.rbtn_RFC = ttk.Radiobutton(self.lbf_modelSelection, text="RFC", variable=self.modelId_gui, value=2,
                                        cursor="hand2")

        self.rbtn_CNN.grid(row=0, column=0, sticky='nw')
        self.rbtn_SVM.grid(row=1, column=0, sticky='nw')
        self.rbtn_RFC.grid(row=2, column=0, sticky='nw')

        ## Part to select the action

        self.lbf_actions = ttk.LabelFrame(self, text="Action")
        self.lbf_actions.pack(side="top", expand=False, fill='both')

        # See performances
        # self.btn_seePerf = ttk.Button(self.lbf_actions, command=self.updateTestResults, text="See performances", cursor="hand2")
        # self.btn_seePerf.pack(side="left", expand=True, fill='both')
        # Execute the model on a given image
        self.btn_runOnImage = ttk.Button(self.lbf_actions, command=self.detect_on_image, text="Run model on image",
                                         cursor="hand2")
        self.btn_runOnImage.pack(side="left", expand=True, fill='both')
        # Execute the model on the webcam
        self.btn_runOnVideo = ttk.Button(self.lbf_actions, command=self.detect_on_videoStream,
                                         text="Run model on webcam", cursor="hand2")
        self.btn_runOnVideo.pack(side="left", expand=True, fill='both')

        # Part with the display

        self.f_results = ttk.Frame(self)
        self.f_results.pack(side="top", expand=True, fill='both', padx=3, pady=2)

        # Part with the display of the test result

        self.lbf_testResult = ttk.LabelFrame(self.f_results, text="Test results")
        self.lbf_testResult.pack(side="left", expand=False, fill='both', padx=3, pady=2)

        self.lbl_result = ttk.Label(self.lbf_testResult, text="...", anchor='w', justify="left", font=self.monoFont)
        self.lbl_result.pack(side='top', expand=True, fill='both', padx=2, pady=2)

        # Part with the image display
        self.canvas1 = tk.Canvas(self.f_results, cursor='plus')
        self.canvas1.pack(side="left", padx=2, pady=2, expand=True, fill='both')

        # Interactions and bindings

        self.modelId_gui.trace_add(mode="write", callback=self.onSelectModel)

        self.modelId_gui.set(0)  # Initialisation à l'affichage du CNN

    def onSelectModel(self, *args):
        self.close_videoStream()
        self.load_model()
        self.updateTestResults()

    def load_model(self, *args):
        self.lbl_result.configure(text="Loading model, please wait...")
        self.close_videoStream()
        print("Loading model...")
        self.process.load_model(self.modelId_gui.get())

    def updateTestResults(self, *args):

        self.close_videoStream()
        text = "Results for model " + self.process.getModelName(self.modelId_gui.get()) + \
               "\n" + self.process.get_training_results()
        self.lbl_result.configure(text=text)

    def detect_on_image(self):

        self.close_videoStream()
        pathToImage = filedialog.askopenfilename(title="Please choose the image file you want to process")
        relativePathToImage = os.path.relpath(pathToImage)
        print("Opening image on path: ", relativePathToImage)

        if len(relativePathToImage) > 0:
            image = self.process.detect_on_image_path(relativePathToImage)

        self.show_image(image)

    def updateFrames(self):
        ret, img = self.capture.read()  # captures frame and returns boolean value and captured image
        if ret and self.loopOnVideoStream:
            processed_img = self.process.detect_on_image(img)
            self.show_image(processed_img)
            MyWindow.after(self.frameDuration, self.updateFrames)
        else:
            self.close_videoStream()

    def close_videoStream(self):
        if self.capture != None:
            print("Closing video stream")
            self.capture.release()
            self.capture = None
            print("Video stream closed")
        self.canvas1.delete('all')

    def detect_on_videoStream(self, streamSourceId=0):

        print("Getting stream source...")
        self.capture = cv.VideoCapture(streamSourceId)
        self.loopOnVideoStream = True
        print("Starting loop")
        self.updateFrames()

    def clearDisplays(self):
        self.lbl_result.configure(text=self.defaultTestResultText)
        self.close_videoStream()
        self.canvas1.delete('all')

    def show_image(self, image):
        """ The input image is in OpenCV format (=numpy array)"""

        self.canvas1.delete('all')
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img_h, img_w = image.shape[0:2]
        ratioWH = img_w / img_h
        w = self.canvas1.winfo_width()
        h = self.canvas1.winfo_height()

        if w / h > 1:
            new_img_h = h
            new_img_w = int(h * ratioWH)
        else:
            new_img_h = int(w / ratioWH)
            new_img_w = w

        image = cv.resize(image, (new_img_w, new_img_h), interpolation=cv.INTER_AREA)

        # convert the images to PIL format...
        image = Image.fromarray(image)

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)

        self.imageDisplayed = image
        image_id = self.canvas1.create_image((0, 0), anchor="nw", image=self.imageDisplayed)
        """
        cv.imshow('Result', image)
        cv.waitKey(0)
        """


# MAIN PROGRAM

# The interface on which the user is working
MyApp = Interface(mainframe=MyWindow, dict_paths=pathDict, emotionsDict=emotionsDict, frameRate=frameRate)
print("Application started")

# Close the GUI
MyWindow.mainloop()