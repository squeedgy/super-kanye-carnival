import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = file.read()

        try:
            #load as JSON
            loaded_data = json.loads(data)

            #use it directly
            if isinstance(loaded_data, dict):
                return [loaded_data]
            else:
                return loaded_data
        except json.JSONDecodeError:
            #assume it's a string if it's not a JSON
            return [{'title': 'Unknown Title', 'artist': 'Unknown Artist', 'lyrics': data, 'view_count': None}]

def clean_lyrics(lyrics):
    #find the index of the word "Lyrics" and keep the text after it
    index = lyrics.find("Lyrics")
    if index != -1:
        cleaned_lyrics = lyrics[index + len("Lyrics"):]
    else:
        cleaned_lyrics = lyrics

    return cleaned_lyrics.strip()

def tokenize_and_remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def preprocess_lyrics(data):
    preprocessed_data = []

    for item in data:
        artist = item.get('artist', 'Unknown Artist')
        title = item.get('title', 'Unknown Title')
        lyrics = item.get('lyrics', '')
        view_count = item.get('view_count', None)

        cleaned_lyrics = clean_lyrics(lyrics)

        #tokenize and remove stopwords
        tokenized_text = tokenize_and_remove_stopwords(cleaned_lyrics)

        #add the preprocessed data
        preprocessed_data.append({
            'title': title,
            'artist': artist,
            'lyrics': tokenized_text,
            'view_count': view_count
        })

    return pd.DataFrame(preprocessed_data)

def save_preprocessed_data(data, output_file):
    data.to_csv(output_file, index=False, encoding='utf-8')

def main():
    #load data from the JSON file
    lyrics_data = load_data('data/raw/kanye_west_lyrics.json')

    #preprocess the data
    preprocessed_data = preprocess_lyrics(lyrics_data)

    #save the preprocessed data to CSV
    output_file = 'data/processed/kanye_lyrics_cleaned.csv'
    save_preprocessed_data(preprocessed_data, output_file)

if __name__ == '__main__':
    main()
