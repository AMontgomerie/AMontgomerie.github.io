---
layout: default
title: "Using Spotipy to Collect Track Data"
date: 2020-07-30
excerpt: For a recent project on classifying music genres, I needed to collect a large dataset of labelled tracks. The Spotify API is ideal for this because, a long with a variety of tabular track data, you can download 30-second track samples from the majority of tracks. An easy way to use the Spotify API in Python is through Spotipy. 

---

<h1>{{ page.title }}</h1>

For a [recent project on classifying music genres](https://github.com/iarfmoose/genre_classifier), I needed to collect a large dataset of labelled tracks. [The Spotify API](https://developer.spotify.com/documentation/web-api/) is ideal for this because, a long with a variety of tabular track data, you can download 30-second track samples from the majority of tracks. An easy way to use the Spotify API in Python is through [Spotipy](https://spotipy.readthedocs.io/en/2.13.0/). 

In this post I'll show how to use Spotipy to get track data, and how to download track samples. We'll also discuss a couple of genre labelling strategies and their issues.

## Using Spotipy

### Setup and Login
The first thing you need to do is register a [Spotify Developer account](https://developer.spotify.com/). From the Dashboard, click "create an app", choose a name, write a description and agree to the terms. This will give you a *Client ID* and a *Client Secret* which you can use to gain access. 

Using Spotipy, we can now log in like this:
```python
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def spotify_login(cid, secret):
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

cid = "<Your Client ID>"
secret = "<Your Client Secret>"

sp = spotify_login(cid, secret)
```

### Getting data

We can now use `sp`'s methods to query Spotify. For example, if I want to find Radiohead, I can call `sp.search()` like this: 
```python
result = sp.search("Radiohead", type='artist')
```
`result` is a dictionary of all the search results. `result['artists']['items']` contains a list of artists, of which the Radiohead we are looking for is the first element. We can extract Radiohead's Artist ID and then use it to find other Radiohead-related data. To get a list of their albums we can query Spotify again, this time calling `sp.artist_albums()` and passing Radiohead's Artist ID as an argument:
```python
id = result['artists']['items'][0]['id']

albums = sp.artist_albums(id)

for album in albums['items']:
    print(album['name'])
```
Which prints the following:
```
OK Computer OKNOTOK 1997 2017
A Moon Shaped Pool
TKOL RMX 1234567
The King Of Limbs
In Rainbows
In Rainbows (Disk 2)
Hail To the Thief
Amnesiac
I Might Be Wrong
Kid A
OK Computer
The Bends
Pablo Honey
Ill Wind
Supercollider / The Butcher
Harry Patch (In Memory Of)
Spectre
Daydreaming
Burn the Witch
The Daily Mail / Staircase
```

### Downloading Track Samples

We can also access 30-second track samples from Spotify using each track's `preview_url` attribute. Note that not all tracks have this enabled. Let's say we want to take the first Radiohead album from the list, which is `'OK Computer OKNOTOK 1997 2017'`, and download track samples from it. We can do this by getting the albums's Album ID and then calling `sp.album_tracks()`. We can then make a list of urls like this:
```python
album_id = albums['items'][0]['id']
album_tracks = sp.album_tracks(album_id)
preview_urls = [track['preview_url'] for track in album_tracks['items']]
```
Now we have the preview urls, we can download them using [urlretrieve](https://docs.python.org/3/library/urllib.request.html):
```python
from urllib.request import urlretrieve

directory = "directory/to/save/in"

for i in range(len(preview_urls)):
  urlretrieve(url, "{}/{}{}".format('directory', 'track{}'.format(i+1), ".mp3"))
```
This will download the numbered tracks into the directory specified.

## Genre Labelling

For my project I needed track samples labelled by genre. Unfortunately Spotify tracks don't have a `genre` attribute so I needed to find a more creative way to label them. I tried two methods for labelling tracks: using playlists as labels, and using artist genres as labels.

### Playlists as Labels

Users often make playlists with a coherent theme, and this theme is sometimes a particular genre. We can use these themed playlists as collections of labelled tracks. Using this method we have a pretty simple data collection strategy: we just need to define a list of genres to search for, search for playlists in each genre, and label any tracks we find with the corresponding genre label.

The following code print the first 10 rock playlists and the number of tracks they contain. Note that in `sp.search()` the maximum `limit` value is 50. If you want to show more results you have to use `offset` to get more pages of results.
```python
def get_playlists_by_genre(genre, limit):
    results = sp.search(genre, limit=limit, type='playlist')
    return results['playlists']['items']

def print_playlist_info(playlists):
    for playlist in playlists:
        print('{}: {}'.format(
            playlist['name'], 
            '{} tracks'.format(playlist['tracks']['total']))
        )

playlists = get_playlists_by_genre('rock', 10)
print_playlist_info(playlists)
```
Which prints:
```
Rock Classics: 150 tracks
Rock This: 50 tracks
Rock Hard: 100 tracks
Rock en Español: 60 tracks
Rock Drive: 100 tracks
Rock & Roll Summer: 71 tracks
Rock Party: 50 tracks
Rock Ballads: 75 tracks
Rock En Espanol 80s 90s 2000s: 85 tracks
Rock Covers: 70 tracks
```
We could replace `print_playlist_info()` with some code containing urlretrieve if we want to download the track data instead. We can also store data about the tracks in a Pandas DataFrame and save it to a CSV file.

#### Issues

An issue with this approach is that we can't guarantee the coherence of the tracks in the playlist. Most Spotify playlists are user-generated, and there is no obligation for tracks to be a single genre, even if the playlist is named in such a way as to indicate that they are.

### Artist Genres as Labels

While individual tracks don't come with genre tags on Spotify, artists do! So instead of relying on the people making playlists to label our data for us, we can just look for artists that have a given genre tag and collect data about their tracks. Unfortunately we can't just do `sp.search('metal', type='artist')` because this will return a list of bands who have *metal* in their band name, not a list of bands that have metal as a genre tag. (Having said this, there does seem to be a bunch of metal bands whose name includes the word "metal"!)

If we can't search for artists by genre, how do find them? One solution is to search by playlist again, and then check each artist in the playlist to see if they have the genre tag we're looking for. If they do then we can collect their music and label it accordingly.

#### Related artists

In addition to this, we can use Spotify's Related Artists feature to find more artists with the same tags. We can just keep iterating recursively through related artists until we can't find any more that have the tags we're looking for. One difficulty with this approach is that related artists are often quite tightly interconnected, meaning that we could end up going round in circles through the same artists endlessly. To solve this, we can build a list of artists as we go: if we already have an artist in our list we can ignore them and stop exploring in that direction, but if we haven't seen them yet we can check their genre tags and delve deeper into their related artists. 

The code for this can be found [here](https://github.com/iarfmoose/genre_classifier/blob/master/metal_subgenre_classifier/spotify_data_collection.ipynb). The recursive part of the code is the following:
```python
def add_related_artists(artist_id, genre_artists, existing_artists):
    if not artist_id:
        return
    related_artists = sp.artist_related_artists(artist_id)['artists']
    for related_artist in related_artists:
        if last_added < search_limit and add_artist(related_artist['id'], genre_artists, existing_artists):
            add_related_artists(related_artist['id'], genre_artists, existing_artists)
```
Given an Artist ID, we query Spotify for their related artists. Then we iterate over the related artists and try adding them to our list. `add_artist()` returns `True` if we are seeing the artist for the first time, in which case it will be added to the list, and `False` otherwise. As long as artists are being added, we keep calling `add_related_artists()` recursively to find more.

#### Issues

However, this approach also has issues. Each artist has a variable length list of genre tags, so it's not clear which is the "correct" label. Artists sometimes make music which crosses genre boundaries, or make several different styles of music over the course of their careers. Since the tags are related to the artist and not to individual tracks, it's difficult to determine exactly which tracks should have which labels.

One solution would be to allow for multiple labels. We could simply add the whole list of labels to each track. But if we want to train a model to classify the tracks, we'll need to build a more complex model which is capable of handling multiple correct labels.

Another solution is to just enforce a one-label-per-artist policy. We can predefine a list of genres that we are looking for, and if we find an artist with that tag, we assign all their music that label. This is a less precise solution for the reasons previously mentioned, and there's also a chance that an artist will fit into several of our chosen classes, so we'd need to be careful to remove any artists that appear multiple times under different genres.

## Conclusion

Using Spotipy, it's fairly simple to collect and save both tabular data and mp3 track samples. It's not so easy to label this data for a genre classification task. In the end, for my project, I decided to go with the simpler solution of enforcing a one-label-per-artist policy. This enabled me to build a multi-class classifier and train it on data which has one correct label per example. The full project repository can be found [here](https://github.com/iarfmoose/genre_classifier).
