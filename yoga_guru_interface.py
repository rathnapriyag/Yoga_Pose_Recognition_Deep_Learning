#Importing packages
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import matplotlib.image as mpimg
from PIL import Image
import pickle
import random
import string
import streamlit as st
import keras
from tensorflow import keras

user_folder = 'C:\\Users\\Priya\\DL_User_Images\\'
cur_dir = os.getcwd()
with open('saved_dictionary_25_yoga.pkl', 'rb') as f:
    class_mapping = pickle.load(f)
main_folder = cur_dir + "\\yoga_basics_data\\dataset\\"
model = keras.models.load_model('cnn_yoga_basics_model_shuffled')

WIDTH=100
HEIGHT=150
st.title("Yoga Guru")
st.text("An interactive assitive program to help you learn basic yoga asanas")
st.header("Basic 25 Asanas")
st.subheader("Program can assist you with the following 25 asanas")
yoga_names = [str(i).title() for i in list(class_mapping.keys())]
asana = st.radio("Pick one asana",yoga_names,index=0)
st.success(asana)
asana=asana.lower()
st.write(asana)



   
def enter_unique_name():
    exiting_file_names=[]
    for root, dirs, files in os.walk(user_folder):
        for f in files:
            exiting_file_names.append(f[:10])
    random_letters_list = random.choices(string.ascii_lowercase, k=10)
    random_letters_string = ''.join(random_letters_list)
    if random_letters_string in exiting_file_names:
        print("Name taken. Enter a different name")
        enter_unique_name()
    else:
        return random_letters_string


def create_single_user_data(image_path):
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (HEIGHT,WIDTH))
    img = np.array(img).astype('float32')
    img = img/255
    return img

def display_single_user_image(img_path):
    image_content = Image.open(img_path)
    st.image(image_content, caption='This is how you look')

def display_asana_samples(asana):
    asana_file_names=[]
    asana_folder = main_folder+str.strip(asana)
    for root, dirs, files in os.walk(asana_folder):
        for f in files:  
            asana_file_names.append(f)
    asana_file_names=random.choices(asana_file_names,k=3)
   
    img_list=[]
    for i in range(len(asana_file_names)):
        img_list.append(Image.open(main_folder+str.strip(asana)+"\\"+asana_file_names[i]))
    st.image(img_list)
    
def predict_user_asana(user_x,asana):
    pred = model.predict(user_x.reshape(1,WIDTH, HEIGHT))
    act_class = class_mapping[asana]
    #act_class = list(class_mapping.keys())[list(class_mapping.values()).index(act_val)]
    pred_val = pred.argmax()
    pred_class = list(class_mapping.keys())[list(class_mapping.values()).index(pred_val)]
    if act_class==pred_val:
        success_msg = "Great job! Model says you are doing "+asana+" the right way!"
        st.success(success_msg)
    else:
        fail_msg = "Sorry..the program says you are doing "+pred_class+" instead...Keep trying!..."
        st.error(fail_msg)   
        
        st.write("Here are some of the sample images for",asana)
        display_asana_samples(asana)
        st.write("Try Again") 
        # else: 
        #     st.write("Uhm..Okay...do it your way...I guess...")
        st.stop()

def individual_attempts(name,asana,attempts):
    cam = cv2.VideoCapture(0)
    val, img = cam.read()
    image_path = user_folder+name +str(attempts)+'.png'
    cv2.imwrite(image_path, img)
    user_x = create_single_user_data(image_path)
    del(cam)
    display_single_user_image(image_path)
    predict_user_asana(user_x,asana)




button = st.button("Perform Yoga")
st.write(button) 
# Create a button, that when clicked, shows a text
if(button):
    name=enter_unique_name()
    st.write(name)
    individual_attempts(name, asana, 1)
    

    


#opening the image




#displaying the image on streamlit app

