import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from .txt_preprocessing import generate_training_and_test_txt
from .utils import calculate_and_print_metrics, calculate_confusion_matrix, save_model_if_better, extract_first_column
from .config import SEED



X_train, X_test, y_train, y_test = generate_training_and_test_txt()

categorical_features = X_train.select_dtypes(include='category').columns.tolist()
txt_features = X_train.select_dtypes(include='object').columns.tolist()
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


categorical_pipeline = Pipeline(
    [
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
    ]
)


txt_pipeline = Pipeline(
    [
        ('extract', FunctionTransformer(extract_first_column, validate=False)),
        ('Tfidf', TfidfVectorizer(max_df=0.7, min_df=0.004))
    ]
)

numerical_pipeline = Pipeline(
    [
        ('standard', StandardScaler())
    ]
)

transformer = ColumnTransformer(
    transformers=[
        ('cat', categorical_pipeline, categorical_features),
        ('txt', txt_pipeline, txt_features),
        ('num', numerical_pipeline, numerical_features)
    ]
)


pipe = Pipeline(
    [
        ('preprocessor', transformer),
        ('classifier', RandomForestClassifier(random_state=SEED,
                                              criterion = 'entropy',
                                              max_depth = 34,
                                              n_jobs=-1,
                                              min_samples_split=0.0001,
                                              n_estimators=55))
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
    y_true = y_train,
    y_pred = y_pred_train,
    type_metric= 'train'
)


calculate_and_print_metrics(
    y_true= y_test,
    y_pred= y_pred_test,
    type_metric= 'test'
)


calculate_confusion_matrix(
    y_true = y_test,
    y_pred = y_pred_test,
    type_metric= 'test'
)



save_model_if_better(
    model = pipe,
    X_test= X_test,
    y_test= y_test,
    name_file= 'model_txt.pkl'
)