import pandas as pd
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def prepare_data(path='data/raw/fake_news/train.tsv'):
    # Завантажуємо
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 
               'barely-true', 'false', 'half-true', 'mostly-true', 'pants-on-fire', 'context']
    df = pd.read_csv(path, sep='\t', names=columns)
    
    # Векторизація
    vectorizer = layers.TextVectorization(max_tokens=10000, output_sequence_length=100)
    vectorizer.adapt(df['statement'].astype(str).values)
    
    # ПЕРЕТВОРЕННЯ В NUMPY (тут ми виправили помилку)
    X = vectorizer(df['statement'].astype(str).values).numpy()
    
    # Мітки (1 - true, 0 - false)
    y = df['label'].apply(lambda x: 1 if x == 'true' else 0).values
    
    return train_test_split(X, y, test_size=0.2, random_state=42)