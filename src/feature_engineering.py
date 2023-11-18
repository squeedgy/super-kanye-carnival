import pandas as pd
from textstat import gunning_fog
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import nltk
nltk.download('vader_lexicon')

def calculate_average_line_length(lyrics):
    #split the lyrics
    lines = lyrics.split('\n')

    #remove duplicate lines
    unique_lines = [line for i, line in enumerate(lines) if i == 0 or line != lines[i - 1]]

    translator = str.maketrans('', '', string.punctuation)
    cleaned_lines = [line.translate(translator).replace(" ", "") for line in unique_lines]

    total_characters = sum(len(line) for line in cleaned_lines)
    total_lines = len(cleaned_lines)

    if total_lines == 0:
        return 0

    average_line_length = total_characters / total_lines

    return average_line_length

def add_sentiment_feature(data):
    sia = SentimentIntensityAnalyzer()

    #add a new column 'sentiment' to the DataFrame
    data['sentiment'] = data['lyrics'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return data

def calculate_gunning_fog_index(lyrics):
    try:
        return gunning_fog(lyrics)
    except Exception as e:
        print(f"Error calculating Gunning Fog Index: {e}")
        return None

def add_gunning_fog_index_feature(data):
    #add a new column 'gunning_fog_index' to the DataFrame
    data['gunning_fog_index'] = data['lyrics'].apply(calculate_gunning_fog_index)
    return data

def calculate_word_count(lyrics):
    words = word_tokenize(lyrics)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return len(filtered_words)

def add_word_count_feature(data):
    #add a new column 'word_count' to the DataFrame
    data['word_count'] = data['lyrics'].apply(calculate_word_count)
    return data

def calculate_free_count(lyrics):
    #tokenize and count the occurrences of the word 'Free'
    words = word_tokenize(lyrics)
    return words.count('Free')

def add_free_count_feature(data):
    #add a new column 'free_count' to the DataFrame
    data['free_count'] = data['lyrics'].apply(calculate_free_count)
    return data

def calculate_most_common_word(lyrics):
    words = word_tokenize(lyrics)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    
    #calculate the most common non-stopword
    word_counter = Counter(filtered_words)
    most_common_word, _ = word_counter.most_common(1)[0]
    
    return most_common_word

def add_most_common_word_feature(data):
    #add a new column 'most_common_word' to the DataFrame
    data['most_common_word'] = data['lyrics'].apply(calculate_most_common_word)
    return data

def calculate_most_common_word_count(lyrics, most_common_word):
    words = word_tokenize(lyrics)
    
    return words.count(most_common_word)

def add_most_common_word_count_feature(data):
    #add a new column 'most_common_word_count' to the DataFrame
    data['most_common_word_count'] = data.apply(lambda row: calculate_most_common_word_count(row['lyrics'], row['most_common_word']), axis=1)
    return data

def add_average_line_length_feature(data):
    #add a new column 'average_line_length' to the DataFrame
    data['average_line_length'] = data['lyrics'].apply(calculate_average_line_length)
    return data

def main():
    # Load the preprocessed data from the CSV file
    input_file = 'data/processed/kanye_lyrics_cleaned.csv'
    data = pd.read_csv(input_file, encoding='utf-8')
    
    data = add_gunning_fog_index_feature(data)
    
    data = add_average_line_length_feature(data)
    
    data = add_sentiment_feature(data)

    data_with_word_count = add_word_count_feature(data)

    data_with_free_count = add_free_count_feature(data_with_word_count)

    data_with_most_common_word = add_most_common_word_feature(data_with_free_count)

    data_with_most_common_word_count = add_most_common_word_count_feature(data_with_most_common_word)

    output_file = 'data/processed/kanye_lyrics_with_features.csv'
    data_with_most_common_word_count.to_csv(output_file, index=False, encoding='utf-8')

if __name__ == '__main__':
    main()