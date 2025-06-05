import os
import numpy as np
import pandas as pd
import pickle
from skimage.io import imread
from skimage.transform import resize, rotate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .config import TEST_SIZE, SEED, SAVE_PATH



def preprocess_tumor_image(img_resized, label, angles, brightness):

    augmented_X = []
    augmented_y  = []
    augmented_images = []

    for angle in angles:
        img_rotated = rotate(img_resized, angle, resize=False, preserve_range=True)
        augmented_images.append(img_rotated)

    img_flipped = np.fliplr(img_resized)
    augmented_images.append(img_flipped)

    for factor in brightness:
        img_bright = np.clip(img_resized * factor, 0, 1)
        augmented_images.append(img_bright)

    for aug_img in augmented_images:
        augmented_X.append(aug_img.flatten())
        augmented_y.append(label)

    return augmented_X, augmented_y



class LoaderFiles:

    def __init__(self):
        pass


    def load_dataset(self, path):

        data = pd.read_csv(path, sep=';')
        return data
    

    def load_images(self, directory, image_size, angles, brightness, for_training=True):

        X, y = [], []
        file_names = []

        if for_training:
        
            for label in os.listdir(directory):

                label_path = os.path.join(directory, label)

                if not os.path.isdir(label_path):
                    continue

                for file in os.listdir(label_path):

                    try:
                        image_path = os.path.join(label_path, file)

                        img = imread(image_path, as_gray=True)
                        img_resized = resize(img, image_size, anti_aliasing=True, preserve_range=True)

                        X.append(img_resized.flatten())
                        y.append(label)

                        aug_X, aug_y = preprocess_tumor_image(img_resized = img_resized,
                                                              label = label,
                                                              angles = angles,
                                                              brightness = brightness)

                        X.extend(aug_X)
                        y.extend(aug_y)

                    except Exception as e:
                        print(f"Error en {image_path}, omitiendo... Error: {e}") # type: ignore


            encoder = LabelEncoder()
            y_encoded= encoder.fit_transform(y)

            save_model(model = encoder, name_file = 'label_encoder.pkl')

            return np.array(X),  y_encoded, encoder
        
        else:

            for file in os.listdir(directory):

                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                try:
                    image_path = os.path.join(directory, file)

                    img = imread(image_path, as_gray=True)

                    X.append(img.flatten())
                    file_names.append(file)
                    
                except Exception as e:
                    print(f"Error en {image_path}, omitiendo... Error: {e}") # type: ignore

            if not X:
                print("No se encontraron imágenes en el directorio.")
                return np.array([]), []

            return np.array(X), file_names
    


class SplitData:

    def __init__(self):
        
        self.test_size = TEST_SIZE
        self.seed = SEED


    def with_shuffling(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = TEST_SIZE,
        shuffle = True,
        random_state = self.seed,
        stratify=y
        )

        return X_train, X_test, y_train, y_test

    
    def without_shuffling(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, stratify=y)

        return X_train, X_test, y_train, y_test



def calculate_and_print_metrics(y_true, y_pred, type_metric):

    print(f"\n\n\nbalanced accuracy {type_metric}: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"precision score {type_metric}: {precision_score(y_true, y_pred, average='micro'):.4f}")
    print(f"recall score {type_metric}: {recall_score(y_true, y_pred, average='micro'):.4f}")
    print(f"f1 score {type_metric}: {f1_score(y_true, y_pred, average='micro'):.4f}")



def calculate_confusion_matrix(y_true, y_pred, type_metric):

    labels = np.unique(y_true)
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    
    index = [f"{label} (Clase Real)" for label in labels]
    columns = [f"{label} (Predicción)" for label in labels]

    df = pd.DataFrame(matrix, index=index, columns=columns)

    print(f"\nConfusion Matrix ({type_metric}):")
    print(df.to_string())
    
    return df



def save_model(model, name_file):

    os.makedirs(SAVE_PATH, exist_ok=True)
    directory = os.path.join(SAVE_PATH, name_file)

    with open(directory, 'wb') as file:
        pickle.dump(model, file)



def load_model(name_file):

    directory = os.path.join(SAVE_PATH, name_file)

    with open(directory, 'rb') as file:

        return pickle.load(file)
    


def compare_models(current_model, best_model, X_test, y_test):

    if best_model is None or current_model.score(X_test, y_test) > best_model.score(X_test, y_test):
        
        return current_model
    
    return best_model



def save_model_if_better(model, X_test, y_test, name_file):

    best_model = None
    directory = os.path.join(SAVE_PATH, name_file)

    if os.path.exists(directory):

        with open(directory, "rb") as file:
            best_model = pickle.load(file)

    best_model = compare_models(model, best_model, X_test, y_test)

    save_model(best_model, name_file)



def extract_first_column(x):
    return x.iloc[:, 0]