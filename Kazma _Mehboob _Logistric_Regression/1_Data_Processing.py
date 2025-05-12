# Import necessary libraries
# Install if missing
# !pip install numpy matplotlib pillow scikit-learn seaborn
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def load_cats_dogs_from_folder(folder_path,image_size=(64,64)):

    images=[]
    labels=[]
    for filename in os.listdir(folder_path):
        try:
            img_path=os.path.join(folder_path,filename)
            img=Image.open(img_path).convert("RGB")
            img=img.resize(image_size)
            img_array=np.array(img).flatten()/255.0  #Normalize Between 0 and 1
            images.append(img_array)


            #Label: 0 for cat , 1 for dog
            if "cat" in filename.lower():
                labels.append(0)
            elif "dog" in filename.lower():
                labels.append(1)
        except Exception as e:
            print(f"Skippimig {filename}: {e}")
            continue   #Skip Unreadable Images
    return np.array(images),np.array(labels)

# Load test images
def load_test_images(folder_path, image_size=(64, 64)):
    images = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(image_size)
            img_array = np.array(img).flatten() / 255.0
            images.append(img_array)
            filenames.append(filename)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue
    
    return np.array(images), filenames


#Load the dataset
test_folder=r"E:\Project\data\archive\test\test_1"
train_folder = r'E:\Project\data\archive\train\train_1'


#Load the images

X_train,y_train=load_cats_dogs_from_folder(train_folder)
X_test,test_filenames=load_test_images(test_folder)

print(f"Traning imagges:{len(X_train)}, Training labels:{len(y_train)}")

print(f"Testing images:{len(X_test)}")


pickle.dump((X_train, X_test, y_train, test_filenames), open("E:\Project\models\split.pkl", "wb"))
      