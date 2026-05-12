import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def prepare_cv_data(csv_path, img_dir, batch_size=32, img_height=180, img_width=180):
    # Завантаження CSV
    df = pd.read_csv(csv_path)
    
    # --- КРИТИЧНО: Очистка імен файлів ---
    df['filename'] = df['filename'].astype(str).str.strip()
    
    # Якщо в CSV немає розширення .jpg, додаємо його автоматично
    if not df['filename'].iloc[0].lower().endswith('.jpg'):
        df['filename'] = df['filename'] + '.jpg'

    # Перевірка (чи бачить Python хоча б один файл?)
    first_file = os.path.join(img_dir, df['filename'].iloc[0])
    print(f"DEBUG: Перевірка шляху -> {first_file}")
    if not os.path.exists(first_file):
        print(f"ПОМИЛКА: Файл не знайдено за шляхом: {first_file}")
    else:
        print("Успіх: Перший файл знайдено!")

    target_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_ds = datagen.flow_from_dataframe(
        dataframe=df,
        directory=img_dir,
        x_col='filename',
        y_col=target_cols,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='raw',
        subset="training",
        seed=123
    )
    
    val_ds = datagen.flow_from_dataframe(
        dataframe=df,
        directory=img_dir,
        x_col='filename',
        y_col=target_cols,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='raw',
        subset="validation",
        seed=123
    )
    
    return train_ds, val_ds