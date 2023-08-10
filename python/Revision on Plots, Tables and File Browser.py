from PyQt4 import QtGui
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

class PrettyWidget(QtGui.QWidget):
    
    
    def __init__(self):
        super(PrettyWidget, self).__init__()
        self.initUI()
        
        
    def initUI(self):
        self.setGeometry(600,300, 1000, 600)
        self.center()
        self.setWindowTitle('Revision on Plots, Tables and File Browser')     
        
        #Grid Layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
                    
        #Canvas and Toolbar
        self.figure = plt.figure(figsize=(15,5))    
        self.canvas = FigureCanvas(self.figure)     
        self.toolbar = NavigationToolbar(self.canvas, self)
        grid.addWidget(self.canvas, 2,0,1,2)
        grid.addWidget(self.toolbar, 1,0,1,2)

        #Empty 5x5 Table
        self.table = QtGui.QTableWidget(self)
        self.table.setRowCount(1)
        self.table.setColumnCount(9)
        grid.addWidget(self.table, 3,0,1,2)
        
        #Import CSV Button
        btn1 = QtGui.QPushButton('Import CSV', self)
        btn1.resize(btn1.sizeHint()) 
        btn1.clicked.connect(self.getCSV)
        grid.addWidget(btn1, 0,0)
        
        #Plot Button
        btn2 = QtGui.QPushButton('Plot', self)
        btn2.resize(btn2.sizeHint())    
        btn2.clicked.connect(self.plot)
        grid.addWidget(btn2, 0,1)
    
        self.show()
    
    
    def getCSV(self):
        filePath = QtGui.QFileDialog.getOpenFileName(self, 
                                                       'Single File',
                                                       '~/Desktop/PyRevolution/PyQt4',
                                                       '*.csv')
        fileHandle = open(filePath, 'r')
        line = fileHandle.readline()[:-1].split(',')
        for n, val in enumerate(line):
            newitem = QtGui.QTableWidgetItem(val)
            self.table.setItem(0, n, newitem)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()    
    
    
    def plot(self):
        y = []
        for n in range(9):
            try:
                y.append(float(self.table.item(0, n).text()))
            except:
                y.append(np.nan)
        plt.cla()
        ax = self.figure.add_subplot(111)
        ax.plot(y, 'r.-')
        ax.set_title('Table Plot')
        self.canvas.draw()
    
    
    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    
        
def main():
    app = QtGui.QApplication(sys.argv)
    w = PrettyWidget()
    app.exec_()


if __name__ == '__main__':
    main()



