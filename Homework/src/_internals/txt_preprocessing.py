import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from .utils import LoaderFiles, SplitData
from .config import CSV_PATH, SEED


loader = LoaderFiles()
splitter = SplitData()
nlp = spacy.load("en_core_web_sm")



def remove_nulls(dataset):
    return dataset.dropna()



def assign_data_types(dataset):

    df = dataset.copy()

    df[['Condition', 'Sex']] = df[['Condition', 'Sex']].astype('category')

    if 'Treatment' in df.columns:
        df['Treatment'] = df['Treatment'].astype('category')

    return df



def copy_clinical_note(dataset):
        
        df = dataset.copy()
        
        df['Clinical_Note_copy'] = df['Clinical Note'].copy()
        df['Clinical_Note_copy'] = df['Clinical_Note_copy'].str.lower()

        return df



def remove_numbers_and_spaces(dataset):
     
    df = dataset.copy()

    #Números
    df['Clinical_Note_copy'] = df['Clinical_Note_copy'].apply(lambda txt: re.sub(r'\d+', ' ', txt))
    #espacios multiples
    df['Clinical_Note_copy'] = df['Clinical_Note_copy'].apply(lambda txt: re.sub(r'\s+', ' ', txt).strip())

    return df



def lemmatize_and_remove_stopwords(txt):

    doc = nlp(txt)

    tokens = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS]

    return ' '.join(tokens)



def clear_data(path, for_training=True):

    df = loader.load_dataset(path = path)
    df = remove_nulls(df)
    df = assign_data_types(df)
    df = copy_clinical_note(df)
    df = remove_numbers_and_spaces(df)
    df['Clinical_Note_copy'] = df['Clinical_Note_copy'].apply(lemmatize_and_remove_stopwords)

    if for_training:

        df = df.sample(frac = 1, random_state = SEED)

        X = df[['Condition', 'Age', 'Sex', 'Clinical_Note_copy']]
        y = df[['Treatment']]

        return X, y
    
    else:
        X = df[['Condition', 'Age', 'Sex', 'Clinical_Note_copy']]
        ids = df['Case ID']
    
        return X, ids



def generate_training_and_test_txt():

    X, y = clear_data(path = CSV_PATH, for_training=True)

    return splitter.without_shuffling(X, y)




if __name__ == '__main__':
    
    generate_training_and_test_txt()