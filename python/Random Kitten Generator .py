import numpy as np
import sys, urllib2
import requests
from PyQt4 import QtGui, QtCore

class PlaceKittenGUI(QtGui.QWidget):
    
    def __init__(self):
        super(PlaceKittenGUI, self).__init__()
        
        self.initUI()
        
    def initUI(self):
        '''Initial UI'''
        
        #Grid Layout
        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)
        
        #Search Button
        self.btn = QtGui.QPushButton('Search', self)
        self.btn.clicked.connect(self.runSearch)      
        self.grid.addWidget(self.btn, 0,2,1,1)
        
        #X Size
        self.x = QtGui.QLineEdit(self)
        self.x.setText('200')
        self.grid.addWidget(self.x, 1,2,1,1)
        
        #Y Size
        self.y = QtGui.QLineEdit(self)
        self.y.setText('200')
        self.grid.addWidget(self.y, 2,2,1,1)
        
        #PLaceholder
        self.label = QtGui.QLabel(self)
        self.label.setText('\t\t\t\t')
        self.grid.addWidget(self.label,2,0,1,1)
        
        #Image
        self.img = QtGui.QLabel(self)
        self.grid.addWidget(self.img, 0,0,2,2)
        
        #Customize Widgets
        self.resize(500, 250)
        self.center()
        self.setWindowTitle('Random Kitty Generator')    
        self.show()
        
    def runSearch(self):
        '''Search Image'''
        x = self.x.text()
        y = self.y.text()
        n = str(np.random.randint(20))
        base = "https://placekitten.com/"
        img_url = base + '{}/{}?image={}'.format(x,y,n)
        data = urllib2.urlopen(img_url).read()
        image = QtGui.QImage()
        image.loadFromData(data)
        self.img.setPixmap(QtGui.QPixmap(image))
        
    def center(self):
        '''Center Widget on screen'''
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
def main():
    '''Codes for running GUI'''
    
    #Create Application object to run GUI
    app = QtGui.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    
    #Run GUI
    gui = PlaceKittenGUI()
    
    #Exit cleanly when closing GUI
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()



