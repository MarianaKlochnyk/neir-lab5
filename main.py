import mlflow
import mlflow.keras
import time
import matplotlib.pyplot as plt

from src.data_loader import prepare_data
from src.model import build_model
from src.utils import save_confusion_matrix


def main():
    # 1. Підготовка даних
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. Створення експерименту
    mlflow.set_experiment("Fake_News_Detection")

    with mlflow.start_run():

        # 3. Створення моделі
        model = build_model(
            vocab_size=10000,
            max_length=100
        )

        # 4. Навчання
        start_time = time.time()

        history = model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32
        )

        duration = time.time() - start_time

        # 5. Графік точності
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['accuracy'], marker='o')
        plt.title('Accuracy of Fake News Model')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.savefig('text_accuracy.png')
        plt.close()

        # 6. Прогнозування
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        # 7. Матриця помилок
        save_confusion_matrix(y_test, y_pred)

        # 8. Логування
        final_accuracy = history.history['accuracy'][-1]

        mlflow.log_metric("accuracy", final_accuracy)
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("text_accuracy.png")
        mlflow.keras.log_model(model, "model")

        # 9. Вивід результатів
        print(f"Час навчання: {duration:.2f} сек")
        print(f"Фінальна точність: {final_accuracy:.4f}")
        print("Графік точності збережено у text_accuracy.png")


if __name__ == "__main__":
    main()