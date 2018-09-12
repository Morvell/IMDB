import numpy as np
from keras.datasets import imdb
from keras.engine.saving import model_from_json
from keras.preprocessing import sequence

if __name__ == '__main__':
    # Устанавливаем seed для повторяемости результатов
    np.random.seed(42)
    # Максимальное количество слов (по частоте использования)
    max_features = 5000
    # Максимальная длина рецензии в словах
    maxlen = 80

    # Загружаем данные
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

    # Заполняем или обрезаем рецензии
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    with open("imdb_model.json", "r") as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("imdb_model.h5")

    model = loaded_model

    # Копмилируем модель
    model.compile(loss='binary_crossentropy',
                  optimizer='ADAM',
                  metrics=['accuracy'])

    # Проверяем качество обучения на тестовых данных
    scores = model.evaluate(X_test, y_test,
                            batch_size=64)
    print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))