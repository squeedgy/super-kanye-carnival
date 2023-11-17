import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
import pandas as pd

def build_text_generation_model(lyrics_data):
    # Preprocess the data
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lyrics_data['lyrics'])

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in lyrics_data['lyrics']:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    #build the model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_text_generation_model(model, X, y, epochs=50):
    #train the model
    model.fit(X, y, epochs=epochs, verbose=1)
    
    model.save('my_model.keras')

def generate_lyrics(model, seed_text, tokenizer, max_sequence_length, next_words=100, output_file='generated_lyrics.txt'):
    #generate new lyrics
    generated_lyrics = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probabilities = model.predict(token_list, verbose=0)[0]
        predicted_index = tf.argmax(predicted_probabilities).numpy()
        output_word = tokenizer.index_word[predicted_index]
        seed_text += " " + output_word
        generated_lyrics += " " + output_word

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(generated_lyrics)

    return seed_text

def preprocess_data_for_training(data):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data['lyrics'])

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in data['lyrics']:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length,
                                                                   padding='pre')

    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    return X, y, tokenizer, max_sequence_length

def train_and_save_model(data):
    X, y, tokenizer, max_sequence_length = preprocess_data_for_training(data)

    model = build_text_generation_model(data)

    train_text_generation_model(model, X, y, epochs=50)

    model.save('trained_model.h5')

    return tokenizer, max_sequence_length

def main():
    #load the preprocessed data from the CSV file
    input_file = 'data/processed/kanye_lyrics_with_features.csv'
    data = pd.read_csv(input_file, encoding='utf-8')

    tokenizer, max_sequence_length = train_and_save_model(data)

    model = load_model('trained_model.h5')

    seed_text = data['lyrics'].iloc[-1]

    generated_lyrics = generate_lyrics(model, seed_text, tokenizer, max_sequence_length, next_words=100)

    print("Generated Lyrics:")
    print(generated_lyrics)

if __name__ == '__main__':
    main()