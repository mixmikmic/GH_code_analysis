# Opens the library file and returns two dictionary objects
def open_library(filename):
    pass

import json

def open_library(filename):
    
    #Create empty dictionaries just in case the library file is empty
    students = {}
    books = {}
    
    #Open the library file encoded in JSON and load it into the data object
    #We use the with keyword so we don't have to explicitly close the file
    #later.
    with open(filename) as f:
        data = json.load(f)
    
    #If there are students or books in the library, 
    #overwrite the empty dictionaries we created
    if data['students'] != {}:
        students = data['students']
    
    if data['books'] != {}:
        books = data['books']
    
    #Return the data we loaded from the file
    return students, books

    #NOTE: The function will return a tuple (students, books)
    #If you want either individually, you can use indexing as shown
    #below in the test section

library = open_library('data/test.json')
students = library[0]
books = library[1]

#\n for whitespace to make reading easier
print(students, '\n\n', books)

def add_book(filename, isbn, title, author):
    #Here's a start
    data = open_library(filename)
    books = data[1]
    
    #Now how can we add books to the data?
    #In the space below, write code that adds the key isbn
    #and the value {'title':title, 'author':author}
    #to the books object.
    
    
    #Finally, write code that writes the new data to the library
    #Do we need to return anything? 
    pass

def remove_book(filename, isbn):
    #See how nicely this works?
    data = open_library(filename)
    books = data[1]
    
    #How can we *remove* an item from a dictionary?
    #Write code to delete the book keyed by isbn in the space below
    
    
    #Now write code that saves the new version of the data to your library
    
    pass

def check_out(filename, isbn, s_id):
    data = open_library(filename)
    books = data[1]
    
    #Find a way to mark a book as checked out. Be sure to associate
    #the book with the student who borrowed it!
    
    
    #And again save the data here
    
    pass

def return_book(filename, isbn):
    data = open_library(filename)
    books = data[1]
    
    #Now ensure that the book is no longer checked out and save the changes
    #to the library.
    
    pass

def status(filename):
    data = open_library(filename)
    books = data[1]
    
    #Print out two lists - one of all books currently checked out,
    #and one of all available books.
    
    pass

