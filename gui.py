from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QFile, QTextStream
import sys
from PyQt5.QtCore import Qt , QEvent 
from PyQt5.QtGui import QPixmap
from time import sleep
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import numpy as np
import cgitb
import glob
global fx,fy,px,py,fx_D,fy_D,px_D,py_D
fx=5064
fy=5735
px=495
py=576
fx_D=400
fy_D=300
px_D=300
py_D=200
cgitb.enable(format = 'text')
class MainWindow(QtWidgets.QMainWindow):
    global fx,fy,px,py,fx_D,fy_D,px_D,py_D
    def __init__(self, *args, **kwargs):
        super().__init__()
        uic.loadUi("gui.ui", self)	
        self.horizontalSlider.setValue(fx)
        self.horizontalSlider_2.setValue(fy)
        self.horizontalSlider_3.setValue(px)
        self.horizontalSlider_4.setValue(py)
        self.horizontalSlider_5.setValue(fx_D)
        self.horizontalSlider_6.setValue(fy_D)
        self.horizontalSlider_7.setValue(px_D)
        self.horizontalSlider_8.setValue(py_D)
        self.horizontalSlider.valueChanged.connect(self.read_slider1)
        self.horizontalSlider_2.valueChanged.connect(self.read_slider2)
        self.horizontalSlider_3.valueChanged.connect(self.read_slider3)
        self.horizontalSlider_4.valueChanged.connect(self.read_slider4)
        self.horizontalSlider_5.valueChanged.connect(self.read_slider5)
        self.horizontalSlider_6.valueChanged.connect(self.read_slider6)
        self.horizontalSlider_7.valueChanged.connect(self.read_slider7)
        self.horizontalSlider_8.valueChanged.connect(self.read_slider8)
        img00 = cv2.imread("F:\\Project\\Bird-eye\\2\\rear.jpeg") 
        img00=cv2.cvtColor(img00,cv2.COLOR_BGR2RGBA)
        self.label_11.setScaledContents(False)
        img00=QImage(img00.data,int(img00.shape[1]),int(img00.shape[0]),QImage.Format_RGBA8888)
        self.label_11.setPixmap((QPixmap.fromImage(img00)).scaled(self.label_11.width(),self.label_11.height(), QtCore.Qt.KeepAspectRatio))
        self.worker1=worker1()
        self.worker1.start()
        self.worker1.imageupdate.connect(self.imageupdateSlot)
        
    def read_slider1(self,value):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.label.setText(str(value))
        fx=value
    def read_slider2(self,value):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.label_2.setText(str(value))
        fy=value
    def read_slider3(self,value):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.label_3.setText(str(value))
        px=value
    def read_slider4(self,value):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.label_4.setText(str(value))
        py=value
    def read_slider5(self,value):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.label_5.setText(str(value))
        fx_D=value
    def read_slider6(self,value):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.label_12.setText(str(value))
        fy_D=value
    def read_slider7(self,value):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.label_14.setText(str(value))
        px_D=value
    def read_slider8(self,value):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.label_15.setText(str(value))
        py_D=value


    
    def imageupdateSlot(self,image):
        # self.label_11.setPixmap((QPixmap.fromImage(img00)).scaled(self.label_11.width(),self.label_11.height(), QtCore.Qt.KeepAspectRatio))
        pic2=image.scaled(self.label_11.width(),self.label_11.height(),Qt.KeepAspectRatio)
        self.label_11.setPixmap(QPixmap.fromImage(pic2).scaled(self.label_11.width(),self.label_11.height(), QtCore.Qt.KeepAspectRatio))
class worker1(QThread):
    imageupdate=pyqtSignal(QImage)
    global mtx,dist
    @QtCore.pyqtSlot()
    def run(self):
        global fx,fy,px,py,fx_D,fy_D,px_D,py_D
        self.threadActive=True
        # dist=np.array([[0.61626835,-2.37821534,0.02727574,0.03075195,-7.06142533]])
        images = glob.glob('F:\\Project\\Bird-eye\\1\\raw_imgs\\jpg0\\*.jpeg')  
        #cv2.namedWindow('calib',cv2.WINDOW_NORMAL)  
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  
    
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)  
        objp = np.zeros((6*7,3), np.float32)  
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) 
        objpoints = [] # 3d point in real world space  
        imgpoints = [] # 2d points in image plane.  
        for fname in images:  
            print(fname)
            img0 = cv2.imread(fname)  
            gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)  
        
            # Find the chess board corners  
            ret, corners = cv2.findChessboardCorners(gray, (7,6),None)  
        
            # If found, add object points, image points (after refining them)  
            if ret == True:  
                objpoints.append(objp)  
        
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)  
                imgpoints.append(corners2)  
        
                
                # Draw and display the corners  
                img000 = cv2.drawChessboardCorners(img0, (7,6), corners2,ret)  
                cv2.namedWindow('calib',cv2.WINDOW_NORMAL)  
                cv2.imshow('calib',img000)  
                cv2.resizeWindow('calib', 600,320)  
                cv2.waitKey(1)  
                
                
        #cv2.destroyAllWindows()  
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)  
        # Printing type of arr object
        print("Array is of type: ", type(dist))
        
        # Printing array dimensions (axes)
        print("No. of dimensions: ", dist.ndim)
        
        # Printing shape of array
        print("Shape of array: ", dist.shape)
        
        # Printing size (total number of elements) of array
        print("Size of array: ", dist.size)
        
        # Printing type of elements in array
        print("Array stores elements of type: ", dist.dtype)
        print(mtx)
        print(dist)
        img = cv2.imread("F:\\Project\\Bird-eye\\2\\rear.jpeg") 
        fx,_,px=mtx[0]
        _,fy,py=mtx[1]
        while self.threadActive:
            # global mtx
            # mtx=np.array([[fx,0,px],[0,fy,py],[0,0,1]],dtype=float)
            mtx[0]=fx,_,px
            mtx[1]=_,fy,py
            mtx[2]=0.0,0.0,1.0
            h,  w = img.shape[:2]   
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))  
            # undistort 
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)  
            # crop the image  
            x,y,w2,h2 = roi  
            #dst = dst[y:y+h, x:x+w]  
            img2=dst[y:y+h2, x:x+w2]  
            # img3=cv2.resize(img2,(w2,h2))  
            img4=cv2.cvtColor(img2,cv2.COLOR_BGR2RGBA )
            img5=QImage(img4.data,int(img4.shape[1]),int(img4.shape[0]),QImage.Format_RGBA8888)
            cv2.imwrite('calibresult.png', img4)
            self.imageupdate.emit(img5)
            print("*",end="",flush=True)
            sleep(0.1)
            

app = QtWidgets.QApplication(sys.argv)

#theme
#file = QFile("Adap/Adap.qss")          #theme1
#file = QFile("Adap//Diff.qss")         #theme2
file = QFile("Adap//Comb.qss")          #theme3
file.open(QFile.ReadOnly | QFile.Text)
stream = QTextStream(file)
app.setStyleSheet(stream.readAll())
#theme


window = MainWindow()
window.show()
app.exec_()