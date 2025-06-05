from .utils import LoaderFiles, SplitData
from .config import IMG_DIR, IMAGE_SIZE, ROTATION_ANGLES, BRIGHTNESS_FACTORS


loader = LoaderFiles()
splitter = SplitData()


def generate_training_and_test_img():

    X, y, encoder = loader.load_images(directory = IMG_DIR,              # type: ignore
                                       image_size = IMAGE_SIZE,
                                       angles = ROTATION_ANGLES,
                                       brightness = BRIGHTNESS_FACTORS,
                                       for_training=True)  

    X_train, X_test, y_train, y_test = splitter.with_shuffling(X, y)

    return X_train, X_test, y_train, y_test, encoder




if __name__ == '__main__':
    
    generate_training_and_test_img()