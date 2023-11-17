import lyricsgenius
import json

GENIUS_API_TOKEN = 'GENIUS_API_TOKEN'

def get_kanye_west_lyrics(genius, song_title):
    #search for specific artist
    song = genius.search_song(song_title, artist="Kanye West")

    if song:
        #return the title, artist, lyrics and view count
        return {
            'title': song.title,
            'artist': song.artist,
            'lyrics': song.lyrics,
            'view_count': song.stats.pageviews if hasattr(song, 'stats') and hasattr(song.stats, 'pageviews') else None
        }
    else:
        print(f"Lyrics not found for the song: {song_title}")
        return None

def save_lyrics_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def main():
    genius = lyricsgenius.Genius(GENIUS_API_TOKEN)

    #specify the list of song
    song_titles = ["Heartless", "Stronger", "Gold Digger", "Jesus Walks", "All of the Lights", "Power", "Love Lockdown", "Runaway", "Famous", "Black Skinhead"]

    #get lyrics for the specified songs
    kanye_lyrics_data = []

    for song_title in song_titles:
        song_data = get_kanye_west_lyrics(genius, song_title)
        if song_data:
            kanye_lyrics_data.append(song_data)

    #save the lyrics
    output_file = 'data/raw/kanye_west_lyrics.json'
    save_lyrics_data(kanye_lyrics_data, output_file)

if __name__ == '__main__':
    main()
