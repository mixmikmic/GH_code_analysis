import numpy as np
from random import randint
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
        self.setWindowTitle('Shironeko Kitty Finder')    
        self.show()
        
    def runSearch(self):
        '''Search Image'''
        
        access = False
        self.setWindowTitle('GENERATING IMAGE')    
        
        while access == False:
            
            year = str(15)
            month = str(np.random.randint(1,12))
            day = str(np.random.randint(1,28))
            pic = str(np.random.randint(1,10))

            value = [year,month,day,pic]

            for i in range(len(value)):
                if len(value[i])==1:
                    value[i] = '0' + value[i] 
            value = ''.join(value)
            base = "http://blog-imgs-79.fc2.com/k/a/g/kagonekoshiro/f"
            img_url = base+value+'.jpg'

            try:
                data = urllib2.urlopen(img_url).read()
                access = True
            except:
                access = False
        
        self.setWindowTitle('Shironeko Kitty Finder')    
        
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
    app.exec_()



if __name__ == '__main__':
    main()



