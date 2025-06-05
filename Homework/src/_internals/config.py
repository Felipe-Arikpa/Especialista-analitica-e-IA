import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.abspath(
    os.path.join(BASE_DIR, '..', '..', '..', 'Input', 'brain_conditions.csv')
)

IMG_DIR = os.path.abspath(
    os.path.join(BASE_DIR, '..', '..', '..', 'Input')
)

SAVE_PATH = os.path.abspath(
    os.path.join(BASE_DIR, '..', '..', '..', 'Output', 'models')
)

IMAGE_SIZE = (128, 128)

ROTATION_ANGLES = [-8, -4, 4, 8]

BRIGHTNESS_FACTORS = [0.85, 1.15]

TEST_SIZE = 0.2

SEED = 666