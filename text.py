import numpy as np
from keras.datasets import imdb
from keras.engine.saving import model_from_json
from keras_preprocessing.text import text_to_word_sequence

if __name__ == '__main__':
    # Устанавливаем seed для повторяемости результатов
    np.random.seed(42)

    word_index = imdb.get_word_index()

    with open("imdb_model.json", "r") as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("imdb_model.h5")

    model = loaded_model

    with open("text.txt", "r") as file:
        text = file.read()
    words = text_to_word_sequence(text)
    test = np.array([word_index[word] if word in word_index and word_index[word] < 88000 else 0 for word in words])

    # Копмилируем модель
    model.compile(loss='binary_crossentropy',
                  optimizer='ADAM',
                  metrics=['accuracy'])

    array = []
    array.append(test)
    a = model.predict(array)

    i = np.around(a[0])[0]

    if i:
        result = "положительна"
    else:
        result = "отрицательна"

    print("Данная рецензия - " + result)