import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from src.data_loader_cv import prepare_cv_data
from src.model_cv import build_cv_model

def plot_results(history):
    """Функція для побудови графіків Accuracy та Loss"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Графік точності
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    # Графік втрат
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png') # Зберігає графік як картинку для звіту
    print("Графіки успішно збережено у файл 'training_results.png'")
    plt.show()

def main():
    # 1. Шляхи до даних
    base_dir = os.path.join('data', 'raw', 'eye_disease')
    csv_path = os.path.join(base_dir, 'full_df.csv')
    img_dir = os.path.join(base_dir, 'preprocessed_images')
    
    print(f"Початок навчання. Шлях до фото: {img_dir}")
    
    # 2. Завантаження даних
    train_ds, val_ds = prepare_cv_data(csv_path, img_dir)
    
    # 3. Налаштування MLFlow та навчання
    mlflow.set_experiment("Eye_Disease_Classification")
    
    with mlflow.start_run(run_name="Final_Model_Run"):
        model = build_cv_model()
        
        # Навчання
        history = model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=5
        )
        
        # 4. Візуалізація результатів
        plot_results(history)
        
        # Логування моделі
        mlflow.keras.log_model(model, "model")
        
        print("--- Навчання завершено! ---")

if __name__ == "__main__":
    main()