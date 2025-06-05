import os
import argparse
import pandas as pd
from ._internals.txt_preprocessing import clear_data
from ._internals.utils import load_model, LoaderFiles, extract_first_column
from ._internals.config import IMAGE_SIZE, ROTATION_ANGLES, BRIGHTNESS_FACTORS



loader = LoaderFiles()


def predict_txt(input_dir, output_dir, model_name='model_txt.pkl'):

    csv_file = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_file:
        print(f'\nNo se encontró archivo CSV en el directorio.')
        
        return

    csv_path = os.path.join(input_dir, csv_file[0]) # type: ignore

    X, case_ids = clear_data(path=csv_path, for_training=False)

    model = load_model(model_name)

    y_pred = model.predict(X)

    result = pd.DataFrame({
        'Case ID': case_ids,
        'Predicted Treatment': y_pred
    })

    output_path = os.path.join(output_dir, 'txt_predictions.csv')
    result.to_csv(output_path, index=False)
    print(f'[historia clínica] Predicciones guardadas en: {output_path}')



def predict_images(input_dir, output_dir, model_name='model_img.pkl', encoder_name='label_encoder.pkl'):

    X, file_names = loader.load_images(directory=input_dir,
                                        image_size=IMAGE_SIZE,
                                        angles = ROTATION_ANGLES,
                                        brightness = BRIGHTNESS_FACTORS,
                                        for_training=False)

    if X.size == 0:
        print(f'\nNo se encontraron imágenes en el directorio.')
        return

    model = load_model(model_name)
    encoder = load_model(encoder_name)

    y_pred = model.predict(X)
    y_labels = encoder.inverse_transform(y_pred)

    result = pd.DataFrame({
        'Image': file_names,
        'Predicted Label': y_labels
    })

    output_path = os.path.join(output_dir, 'image_predictions.csv')
    result.to_csv(output_path, index=False)
    print(f'[Imágenes] Predicciones guardadas en: {output_path}')



def run_predictions(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    predict_txt(input_dir=input_dir, output_dir=output_dir)
    predict_images(input_dir=input_dir, output_dir=output_dir)



def main():

    parser = argparse.ArgumentParser(description='Ejecuta predicciones sobre texto e imágenes.')
    parser.add_argument('--input_dir', required=True, help='Directorio de entrada con el CSV y las imágenes.')
    parser.add_argument('--output_dir', required=True, help='Directorio donde se guardarán las predicciones.')

    args = parser.parse_args()
    run_predictions(args.input_dir, args.output_dir)



if __name__ == '__main__':

    main()