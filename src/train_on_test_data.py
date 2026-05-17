import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split

# Supress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

INPUT_SIZE = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTING_DIR = os.path.join(BASE_DIR, '..', 'Testing')

def load_binary_data():
    dataset = []
    labels = []
    
    no_dir = os.path.join(TESTING_DIR, 'Binary', 'no')
    yes_dir = os.path.join(TESTING_DIR, 'Binary', 'yes')
    
    # Load No Tumor
    if os.path.exists(no_dir):
        for img_name in os.listdir(no_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(no_dir, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((INPUT_SIZE, INPUT_SIZE))
                dataset.append(np.array(img))
                labels.append(0)
                
    # Load Tumor
    if os.path.exists(yes_dir):
        for img_name in os.listdir(yes_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(yes_dir, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((INPUT_SIZE, INPUT_SIZE))
                dataset.append(np.array(img))
                labels.append(1)
                
    return np.array(dataset, dtype=np.float32) / 255.0, np.array(labels)

def load_multiclass_data():
    dataset = []
    labels = []
    
    classes = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
    
    for cls_name, cls_idx in classes.items():
        cls_dir = os.path.join(TESTING_DIR, 'MultiClass', cls_name)
        if os.path.exists(cls_dir):
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(cls_dir, img_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = img.resize((INPUT_SIZE, INPUT_SIZE))
                    dataset.append(np.array(img))
                    labels.append(cls_idx)
                    
    return np.array(dataset, dtype=np.float32) / 255.0, np.array(labels)

def train_models():
    # 1. Train Binary Model
    print("Loading binary data...")
    X_bin, y_bin = load_binary_data()
    print(f"Loaded {len(X_bin)} binary images.")
    
    if len(X_bin) == 0:
        print("No binary data found. Skipping.")
    else:
        y_bin_cat = keras.utils.to_categorical(y_bin, num_classes=2)
        
        # Load the model we built before
        model_path = os.path.join(BASE_DIR, 'epoch10_sgd_acc96Point76.h5')
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}...")
            model = keras.models.load_model(model_path)
            
            # Train on the small dataset
            # We need to replicate the input 3 times for the ensemble
            print("Training binary model...")
            model.fit([X_bin, X_bin, X_bin], y_bin_cat, epochs=20, batch_size=4)
            
            model.save(model_path)
            print(f"Saved updated binary model to {model_path}")
        else:
            print(f"Model not found at {model_path}. Run build_models.py first.")

    # 2. Train Multiclass Model
    print("\nLoading multiclass data...")
    X_multi, y_multi = load_multiclass_data()
    print(f"Loaded {len(X_multi)} multiclass images.")
    
    if len(X_multi) == 0:
        print("No multiclass data found. Skipping.")
    else:
        y_multi_cat = keras.utils.to_categorical(y_multi, num_classes=3)
        
        model_path = os.path.join(BASE_DIR, 'multi-model-30K-epouch20.h5')
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}...")
            model = keras.models.load_model(model_path)
            
            print("Training multiclass model...")
            model.fit([X_multi, X_multi, X_multi], y_multi_cat, epochs=20, batch_size=4)
            
            model.save(model_path)
            print(f"Saved updated multiclass model to {model_path}")
        else:
            print(f"Model not found at {model_path}. Run build_models.py first.")

if __name__ == '__main__':
    train_models()
