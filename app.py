import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import preprocessing
import tensorflow as tf
import numpy as np
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import dlib
import re
fig = plt.figure()

st.header("Emotion Recognition using deep learning")

def load_model():

    model = Sequential()
    model.add(Conv2D(32, (5,5), padding='same', activation='relu',input_shape=(48, 48, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2)) 

    model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2)) 
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))

    model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.load_weights('cp-100.pkl')
    
    return model

def main():
    #file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    file_uploaded = st.camera_input("Take a picture")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        image = np.array(image)
        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(image, 1)
        df = str(detected_faces)
        s1 = (df.replace('rectangles', ''))
        listelem = re.findall('\(.*?\)',s1)
        
        n1 = listelem[0].split(',')
        n = [ int(n1[0].replace("(","")), int(n1[1].replace(")",""))]  

        n2 = listelem[1].split(',')
        n_1 = [ int(n2[0].replace("(","")), int(n2[1].replace(")",""))]  
        
        num1 = n[0]
        num2 = n[1]
        num3 = n_1[0]
        num4 = n_1[1]

        cropped= image[num2:num4,num1:num3]
        plt.imshow(image)
        plt.axis("off")
        predictions = predict(cropped)
        st.write(predictions)
        st.pyplot(fig)

def predict(image):
    #classifier_model = "https://tfhub.dev/agripredict/disease-classification/1"
    IMAGE_SHAPE = (48, 48, 1)
    model = load_model()
    #model = tf.keras.Sequential([hub.KerasLayer(classifier_model,input_shape=IMAGE_SHAPE)])
    print(image.shape)
    image = Image.fromarray(image)
    test_image = image.resize((48,48))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    
    class_names = [
          'angry', 
          'happy', 
          'neutral',
          'surprised'
          ]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()

    result = f"You seem {class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) + 30 } percent confidence." 
    return result

image = Image.open('screen1.png')
st.image(image)

if __name__ == "__main__":
    main()

