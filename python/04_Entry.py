from tkinter import *
root = Tk()
root.title("Basic Entry Widget")

ent = Entry(root)
ent.pack()

def show_data():
    print(ent.get())
Button(root,
      text="Show Data",
      command=show_data).pack()

root.mainloop()



