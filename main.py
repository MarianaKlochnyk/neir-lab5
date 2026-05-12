import mlflow
import mlflow.keras
import time
from src.data_loader import prepare_data
from src.model import build_model
from src.utils import save_confusion_matrix

def main():
    # 1. Готуємо дані
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 2. MLflow експеримент
    mlflow.set_experiment("Fake_News_Detection")
    
    with mlflow.start_run():
        # Створюємо модель
        model = build_model(vocab_size=10000, max_length=100)
        
        # Навчання
        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=5, batch_size=32)
        duration = time.time() - start_time
        
        # Оцінка
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        save_confusion_matrix(y_test, y_pred)
        
        # Логування (як того вимагає лаба)
        mlflow.log_metric("accuracy", history.history['accuracy'][-1])
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.keras.log_model(model, "model")
        
        print(f"Експеримент завершено! Точність: {history.history['accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()