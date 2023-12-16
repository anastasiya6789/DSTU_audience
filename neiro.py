from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Подготовка данных
with open('dataset.txt', 'r', encoding='cp1251') as file:
    mat_words = [word.strip() for word in file]

non_mat_words = ["example", "non-mat", "words", "new-word-1", "new-word-2"]
X_train = mat_words + non_mat_words  # Объединяем матерные и нематерные слова
y_train = [1] * len(mat_words) + [0] * len(non_mat_words)  # Метки для матерных и нематерных слов

vectorizer = CountVectorizer()  # Создаем объект для векторизации текста
X_train_vectorized = vectorizer.fit_transform(X_train)  # Векторизуем обучающий набор слов

# Обучение модели
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


# Проверка наличия матерных слов
def check_word(word):
    word = word.lower()  # Приводим слово к нижнему регистру

    if word not in vectorizer.vocabulary_:
        return "Слово не матерное"

    word_vectorized = vectorizer.transform([word])  # Векторизуем вводимое слово
    prediction = model.predict(word_vectorized)  # Предсказываем метку
    if prediction[0] == 1:
        return "Слово матерное"
    else:
        return "Слово не матерное"


