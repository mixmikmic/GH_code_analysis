# Imports
# ---------
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# plt.style.use('default')
# print(plt.rcParams)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import numpy as np

import tkinter as tk



#################################
# Print formatting for np.arrays.
np.set_printoptions(formatter={'float':'{: 0.4f}'.format})
    

def calculate(history, state, count):
    
    # Seed
    y_0 = 0.3 
    
    # History array.
    # The first index corresponds to the order in which the plots where viewed.
    # The second index - 0-r_max_var, 1-r_min_var, 2-y_max_var, 3-y_min_var,
    # 4-n, 5-m, 6-ph, 7-pw
    
    # Getting paramters
    r_max = float(history[state][0])
    r_min = float(history[state][1])
    y_max = float(history[state][2])
    y_min = float(history[state][3])
    n = int(history[state][4])
    m = int(history[state][5])
    ph = int(history[state][6])
    pw = int(history[state][7])

    # print (r_max, type(r_max))
    # print (r_min, type(r_min))
    # print (y_max, type(y_max))
    # print (y_min, type(y_min))
    # print (n, type(n))
    # print (m, type(m))
    # print (ph, type(ph))
    # print (pw, type(pw))
    
    # Use the following for calculating and displaying n_fsd
    # n_fsd = width of plot in pixels. i.e, 1 fsd per pixel   
    n_fsd = pw
    n_fsd_var.set(n_fsd)
    
    # Creating an array r with n_fsd elements, spaced equally between r_min and r_max.
    r = np.linspace(r_min, r_max, n_fsd)
    # print(r.shape)
    # print (r)
    
    # Creating a 2D array bd with shape (len(r),m)
    bd = np.zeros((len(r),(m)))
    # print(bd.shape)
    
    # Calculating values
    # ------------------
    # This is the only place where the equation is present. Change it to obtain
    # other bifurcation diagrams. Change x-y limits of plots to see graph.
    i = 0    # Iterate to keep track of the fsds. Each i corresponds to 1 fsd.
    # For each dr, calculating the final state diagram.
    for dr in r:
        temp = y_0
        # Calculating but not storing the first n values.
        for j in range(0, n):
            y_t = temp * dr * (1 - temp)
            temp = y_t
        # The first element will be the (n+1)-th iterate.    
        # Calculating and storing the next m values.
        bd[i][0] = y_t
        for k in range(1, m):
            bd[i][k] = bd[i][k-1] * dr * (1 - bd[i][k-1])
        # Updating i value
        i = i + 1

    # Plotting
    # --------
    
    # Setting new dimensions(in inches)
    f.set_size_inches((pw/100, ph/100))
    # Setting the new canvas size (in pixels)
    canvas.get_tk_widget().config(width=pw, height=ph)
    ### FIGURE OUT WHICH ONE OF THE ABOVE IS REQUIRED OR IF BOTH ARE

    # Clearing existing figure (otherwise it builds up in memory.)
    plt.clf()

    # Plotting from data.
    plt.plot(r, bd, color='black', linestyle='None', marker='.', markersize=1.5)
    plt.title('Bifurcation Diagram')
    plt.xlabel('$r$')
    plt.ylabel('$x$')
    plt.xlim(r_min, r_max)
    plt.ylim(y_min, y_max)

    # This function draws the plot onto the canvas.
    plt.gcf().canvas.draw()
    
    # Declare and register callbacks for getting axes limits 
    # when zooming using the matplotlib toolbar.
    def on_xlims_change(axes):
        r_min_var.set(axes.get_xlim()[0])
        r_max_var.set(axes.get_xlim()[1])
        # Storing the history for the zoomed plot.
        # Only indices 0, 1, 2, and 3 need be changed.
        history[count.get()]=history[count.get()-1]
        history[count.get()][0]=r_max_var.get()
        history[count.get()][1]=r_min_var.get()
      
    def on_ylims_change(axes):
        y_min_var.set(axes.get_ylim()[0])
        y_max_var.set(axes.get_ylim()[1])
        history[count.get()][2]=y_max_var.get()
        history[count.get()][3]=y_min_var.get()
        # Upating the listbox
        listbox.insert(tk.END, count.get())
        # Incrementing the count
        count.set(count.get()+1)
        state = count.get() - 1
        # When zooming with the toolbar, it loses resolution
        # as it is digital zooming.
        # To maintain the mentioned n_fsd, the calculate function called.
        calculate(history, state, count)
        # print ("updated ylims: "+ str(axes.get_ylim()))
        
    # Upon detecting zoom both the functions are called.
    # It is assumed that zoom along only 1 axis is not possible
    # with the rectangle zoom tool.
    # count needsto updated in one funcyion only.
    plt.gca().callbacks.connect('xlim_changed', on_xlims_change)
    plt.gca().callbacks.connect('ylim_changed', on_ylims_change)
    
    
# Update function. Collects the values from the fields
# and calls the calculate function.
def update(history):
    
    # The update button is meant to be clicked after all the
    # parameters are set to the required value. Hence only reading
    
    ph = ph_var.get()
    pw = pw_var.get()
#     print(type(ph_var.get()))
#     print(ph_var.get())

    
    # Getting other paramters
    r_min_in = r_min_var.get()
    r_max_in = r_max_var.get()
    y_min_in = y_min_var.get()
    y_max_in = y_max_var.get()
    n_in = n_var.get()
    m_in = m_var.get()
        
    # Inputting an index for the current plot into the listbox
    # at the end of the list.
    listbox.insert(tk.END, count.get())
    # Storing the information about plot into the history array.
    # The first index corresponds to the index of the plot.
    # The second index - 0-r_max, 1-r_min, 2-y_max, 3-y_min,
    # 4-n, 5-m, 6-ph, 7-pw
    history[count.get()]=[r_max_in, r_min_in, y_max_in, y_min_in, n_in, m_in, ph, pw]
    # print(history[count.get()])
    
   
    # Increasing count of history states by 1
    # Incrementing here as it could possibly be incremented
    # inside the calcuate function (while zooming using the toolbar).
    count.set(count.get()+1)
    # Calling plot_state function and passing arguments
    state = count.get() - 1
    # state is used for index of plot for plotting.
    # The calculate function calculates and plots the info.
    calculate(history, state, count)
    
def plot_state(history):
    # Getting the current selection when the button plot_history was clicked.
    state = int(listbox.curselection()[0])
    
    # Setting the widgets to display the current state.
    r_max_var.set(float(history[state][0]))
    r_min_var.set(float(history[state][1]))
    y_max_var.set(float(history[state][2]))
    y_min_var.set(float(history[state][3]))
    n_var.set(int(history[state][4]))
    m_var.set(int(history[state][5]))
    ph_var.set(int(history[state][6]))
    pw_var.set(int(history[state][7]))
    
    # The calculate function calculates and plots the info.
    calculate(history, state, count)

        
### PROGRAM STARTS EXECUTING HERE ###
###            MAIN               ###

# Creating root window.
root = tk.Tk() 
# Setting window title.
root.title("Bifurcation Diagram Plotter")

# Creating a frame for all the widgets, except canvas
frame = tk.Frame(root)
frame.grid(row=0, column=1, padx=20, pady=20)

# Defining tkinter variables.
r_max_var = tk.DoubleVar()
r_min_var = tk.DoubleVar()
y_max_var = tk.DoubleVar()
y_min_var = tk.DoubleVar()
n_fsd_var = tk.IntVar()
n_var = tk.IntVar()
m_var = tk.IntVar()
ph_var = tk.IntVar()
pw_var = tk.IntVar()
# Variable to keep track of history
count = tk.IntVar()    
# Initiatilzing
r_max_var.set(4.0)
r_min_var.set(0.0)
y_max_var.set(1.0)
y_min_var.set(0.0)
n_fsd_var.set(700)
n_var.set(100)
m_var.set(100)
ph_var.set(500)
pw_var.set(700)
count.set(0)

# The history array
# The first index corresponds to the order in which the plots where viewed. (max-50)
# The second index - 0-r_max_var, 1-r_min_var, 2-y_max_var, 3-y_min_var,
# 4-n, 5-m, 6-ph, 7-pw
history = np.zeros((50, 8))

# Text entry widgets.
tk.Label(frame, text="Plot Paramteres").grid(row=0, column=0, columnspan=2)
tk.Label(frame, text="r minimum:").grid(row=1, column=0)
r_min_entry = tk.Entry(frame, justify='center', textvariable=r_min_var)
r_min_entry.grid(row=1,column=1)
tk.Label(frame, text="r maximum:").grid(row=2, column=0)
r_max_entry = tk.Entry(frame, justify='center', textvariable=r_max_var)
r_max_entry.grid(row=2,column=1)
tk.Label(frame, text="y minimum:").grid(row=3, column=0)
y_min_entry = tk.Entry(frame, justify='center', textvariable=y_min_var)
y_min_entry.grid(row=3,column=1)
tk.Label(frame, text="y maximum:").grid(row=4, column=0)
y_max_entry = tk.Entry(frame, justify='center', textvariable=y_max_var)
y_max_entry.grid(row=4,column=1)
# The Number of FSD is dynamically calculated according plot width (in update function)
tk.Label(frame, text="Number of FSD:").grid(row=5, column=0)
n_fsd_out = tk.Label(frame, textvariable=n_fsd_var)
n_fsd_out.grid(row=5, column=1)
tk.Label(frame, text="Number of iterates to skip (n):").grid(row=6, column=0)
n_entry = tk.Entry(frame, justify='center', textvariable=n_var)
n_entry.grid(row=6,column=1)
tk.Label(frame, text="Number of iterates to plot (m):").grid(row=7, column=0)
m_entry = tk.Entry(frame, justify='center',textvariable=m_var)
m_entry.grid(row=7,column=1)
tk.Label(frame, text="Plot Height (px):").grid(row=8, column=0)
ph_entry = tk.Entry(frame, justify='center', textvariable=ph_var)
ph_entry.grid(row=8,column=1)
tk.Label(frame, text="Plot Width (px):").grid(row=9, column=0)
pw_entry = tk.Entry(frame, justify='center', textvariable=pw_var)
pw_entry.grid(row=9,column=1)

# create button. Calls the update function when pressed.
update_plot = tk.Button(frame, text="Create Plot", command=lambda: update(history))
update_plot.grid(row=10, column=0, columnspan=2, sticky='EWNS')

# An empty label for space.
tk.Label(frame).grid(row=11, column=0, sticky='NEWS')

# Listbox definition
# listbox_frame = tk.Frame(root)
# listbox_frame.grid(row=2, column=1, sticky="NSEW")
tk.Label(frame, text="Plot History"). grid(row=12, column=0, columnspan=2)
listbox = tk.Listbox(frame)
listbox.grid(row=13, column=0, columnspan=2, sticky="NEWS")
plot_history = tk.Button(frame, text="Plot selection", command=lambda:plot_state(history))
plot_history.grid(row=14, column=0, columnspan=2, sticky="NEWS")

# Initializing figure
f = plt.figure(figsize=(7, 5), dpi=100)

# tk Canvas Widget
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().grid(row=0, column=0)
canvas.get_tk_widget().config(width=0, height=0)




# The Matplotlib toolbar
toolbar_frame = tk.Frame(root)
toolbar_frame.grid(row=1 , column=0, columnspan=2, sticky = 'NEWS')
toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
toolbar.update()
# canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# Calling the root mainloop
root.mainloop()



