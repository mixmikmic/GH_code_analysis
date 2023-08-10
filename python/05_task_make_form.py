from tkinter import *

fields = 'First Name', 'Last Name', 'Country', 'State', 'City', 'Mobile Number'


def fetch(entries):
    for entry in entries:
        field = entry[0]
        text = entry[1].get()
        print("%s:%s"%(field, text))

def makeform(root, fields):
    entries = []
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=15, text=field, anchor='w')
        ent = Entry(row)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((field, ent))
    return entries

if __name__ == '__main__':
    root = Tk()
    ents = makeform(root, fields)
    
    b_show = Button(root,
                   text='Show',
                   command=(lambda e=ents: fetch(ents))
                   )
    b_stop = Button(root,
                   text='Quite',
                   command=root.destroy)

    b_show.pack(side=LEFT, padx=5, pady=5)
    b_stop.pack(side=LEFT, padx=5, pady=5)

    root.mainloop()



