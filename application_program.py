##
# @file affichage_image.py
# Principal file containing the python program
# @author Léo-Emmanuel JOINT (4A GPSE)
# @date July 2022

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
# In the constructor of the Interface class, the GUI elements are not documented, except for the control variable


# Program that shows the GUI

import numpy as np
##
# @brief To generate the GUI
import tkinter as tk
##
# @brief Package adding styles to the GUI elements, as well as other elements
import tkinter.ttk as ttk
##
# @brief To generate a file selection window
from tkinter import filedialog
##
# @brief To manipulate images, and convert them into a version compatible with Tkinter
from PIL import Image, ImageTk
##
# @brief To execute the image processing
import cv2 as cv
##
# @brief To manipulate arrays
import numpy as np
##
# @brief To navigate and search through the directories
import os
# @brief To measure the duration (for testing purposes)
import time # For testing purposes
# @brief To work with excel files
import pandas as pd
# @brief To work with timestamps
from datetime import datetime

# Root window
MyWindow = tk.Tk()

# STYLE

##
# Theme to use in the interface (sets a style for each gui element)
# Not all themes are avaliable depending on the distribution (Windows, Linux or Mac os)
s1 =ttk.Style()
#print("Available themes:", s1.theme_names()) #('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative') on Windows 10
#print("Theme in use:", s1.theme_use()) # vista
s1.theme_use('clam') # Sets the theme

##
# Font for the species displayed (italic)
titleFont = ('Helvetica', 16, 'bold')

# Paths to the models

pathModels = "Models"
pathModel_weights_ConvNet = "FAA_project/Model_CNN_V1/model_convnet.h5"
pathModel_SVM = ""
pathModel_RFC = ""

repo_path = ""

# CLASSES

class FER_Model:
    
    def __init__(self, name=""):

        # Name of the model
        self.name = name

        # List that contains all the classes
        self.dict_classes = {
            0: 'angry',
            1: 'disgusted',
            2: 'fearful',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprised'}


        print("Importing model...")
        if name == "ConvNet":
            pass
        if name == "SVM":
            pass
        if name == "RFC":
            pass
    
    def get_details(self):
        pass
    
    def get_performances(self):
        pass
    
    def getPrediction(self, image):
        """ The input image size should be 48*48 pixels
        Returns the name of the class detected"""
        pass

class Processing:

    def __init__(self, pathData=""):
        pass
    
    def get_model(self, model):
        pass
    
    def detect_expression(self, image, model=0):
        pass
    
    def detect_on_image(self, image_path, model=0):
        image = cv.imread(image_path)
        return(image)
    
    def detect_on_video(self, video, model=0):
        pass
        
    def get_training_results(self, model=0):
        pass

    def get_dataset_details(self):
        pass
    

class Interface(ttk.Frame):
    """ Class that manages all the interface elements """

    def __init__(self, mainframe):
        ttk.Frame.__init__(self, master=mainframe)
        self.pack(side='top', expand=True, fill='both')
        self.master.title("Face Emotion Recognition test program")

        ## VARIABLES

        self.modelId_gui = tk.IntVar(value=1)

        self.process = Processing()

        ## USER INTERFACE

        ### Selection part for the model

        self.lbf_modelSelection = ttk.LabelFrame(self, text="Model selection")
        self.lbf_modelSelection.pack(side='top', expand=False, fill='both', anchor='w', padx=3, pady=2)

        self.rbtn_model1 = ttk.Radiobutton(self.lbf_modelSelection, text="Model 1: ConvNet", variable=self.modelId_gui, value=1, cursor="hand2")
        self.rbtn_model2 = ttk.Radiobutton(self.lbf_modelSelection, text="Model 2: linear SVM", variable=self.modelId_gui, value=2, cursor="hand2")
        self.rbtn_model3 = ttk.Radiobutton(self.lbf_modelSelection, text="Model 3: Random Forest Classifier", variable=self.modelId_gui, value=3, cursor="hand2")

        self.rbtn_model1.grid(row=0, column=0, sticky='nw')
        self.rbtn_model2.grid(row=1, column=0, sticky='nw')
        self.rbtn_model3.grid(row=2, column=0, sticky='nw')

        ## Part to select the action

        self.lbf_actions = ttk.LabelFrame(self, text="Action")
        self.lbf_actions.pack(side="top", expand=False, fill='x', padx=3, pady=2)

        # See performances
        self.btn_seePerf = ttk.Button(self.lbf_actions, text="See performances", cursor="hand2")
        self.btn_seePerf.pack(side="left", expand=True, fill='both')
        # Execute the model on a given image
        self.btn_runOnImage = ttk.Button(self.lbf_actions, command=self.detect_on_image, text="Run model on image", cursor="hand2")
        self.btn_runOnImage.pack(side="left", expand=True, fill='both')
        # Execute the model on a given video
        self.btn_runOnVideo = ttk.Button(self.lbf_actions, command=self.detect_on_video, text="Run model on video", cursor="hand2")
        self.btn_runOnVideo.pack(side="left", expand=True, fill='both')

        # Part with the display of the result

        self.lbf_result_display = ttk.LabelFrame(self, text="Display")
        self.lbf_result_display.pack(side="top", expand=True, fill='both', padx=3, pady=2)

        self.lbl_result = ttk.Label(self.lbf_result_display, text="Results: \n")
        self.lbl_result.pack(side='top', padx=2, pady=2)

        self.canvas1 = tk.Canvas(self.lbf_result_display, cursor='plus')
        self.canvas1.pack(side="top", padx=2, pady=2, expand=True, fill='both')

    def detect_on_image(self):
        pathToImage = filedialog.askopenfilename(title="Please choose the image file you want to process")
        print(pathToImage)
        if len(pathToImage) > 0 :
            image = self.process.detect_on_image(pathToImage, model=self.modelId_gui.get())
        self.show_image(image)

    def detect_on_video(self):
        pathToVideo = filedialog.askopenfilename(title="Please choose the video file you want to process")
        print(pathToVideo)
        if len(pathToVideo) > 0 :
            self.process.detect_on_Video(pathToVideo, model=self.modelId_gui.get())

    def show_image(self, image):
        """ The input image is in OpenCV format (=numpy array)"""
        """
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image = Image.fromarray(image)

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)

        image_id = self.canvas1.create_image((0, 0), anchor="nw", image=image)
        """
        cv.imshow('Result', image)
        cv.waitKey(0)
        print("Image displayed")
        pass



# MAIN PROGRAM

# The interface on which the user is working
MyApp = Interface(mainframe=MyWindow)

# Close the GUI
MyWindow.mainloop()