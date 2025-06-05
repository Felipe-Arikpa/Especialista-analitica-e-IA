import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import umap
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from .img_preprocessing import generate_training_and_test_img
from .utils import calculate_and_print_metrics, calculate_confusion_matrix, save_model_if_better
from .config import SEED



X_train, X_test, y_train, y_test, encoder = generate_training_and_test_img()


pipe = Pipeline(
    [
        ('reducer', umap.UMAP(random_state = SEED,
                              n_neighbors = 8,
                              n_components = 30,
                              min_dist = 0.04,
                              metric = 'manhattan',
                              spread = 1.2,
                              local_connectivity = 0.7)),
        ('classifier', MLPClassifier(solver = 'adam',
                                    early_stopping = True,
                                    learning_rate = 'adaptive',
                                    random_state = SEED,
                                    hidden_layer_sizes = (90, 30),
                                    alpha = 0.03,
                                    tol = 0.00000001,
                                    batch_size = 32,
                                    n_iter_no_change = 20))
    ]
)

pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)



calculate_and_print_metrics(
    y_true= y_train,
    y_pred= y_pred_train,
    type_metric= 'train'
)


calculate_confusion_matrix(
    y_true = encoder.inverse_transform(y_train),
    y_pred = encoder.inverse_transform(y_pred_train),
    type_metric= 'train'
)


calculate_and_print_metrics(
    y_true= y_test,
    y_pred= y_pred_test,
    type_metric= 'test'
)


calculate_confusion_matrix(
    y_true = encoder.inverse_transform(y_test),
    y_pred = encoder.inverse_transform(y_pred_test),
    type_metric= 'test'
)


save_model_if_better(
    model = pipe,
    X_test= X_test,
    y_test= y_test,
    name_file= 'model_img.pkl'
)