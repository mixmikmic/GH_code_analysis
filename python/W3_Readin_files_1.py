lyrics = "Imagine all the people, \nliving life in peace... \n\tJohn Lennon"
print(lyrics)

our_file = open("imagine_lyrics.txt","w")

our_file.write("Imagine by John Lennon")

our_file.write(lyrics)

our_file.close()

lyrics_imported = open("imagine_lyrics.txt","r")

print(lyrics_imported)

lyrics_imported.read()

print lyrics_imported.read()

lyrics_imported.readlines()

lyrics_imported = open("imagine_lyrics.txt","r")

lyrics_imported.readlines()

lyrics_imported = open("imagine_lyrics.txt","r")
lyrics_list = lyrics_imported.readlines()

type(lyrics_list)

lyrics_list[0]

lyrics_list[1]

lyrics_imported.close()

with open("imagine_lyrics.txt","r") as data_imported:
    data = data_imported.readlines()

print data

print data_imported

