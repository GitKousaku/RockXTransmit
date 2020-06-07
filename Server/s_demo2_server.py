#coding:Shift-JIS
import face_reco3x0 as face_reco
import pyscreenshot as ImageGrab
import numpy as np
import cv2
import tkinter
import PIL.Image,PIL.ImageTk
import sys
import os
#from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, 
#                             QTextEdit, QGridLayout, QApplication, QPushButton,  QDesktopWidget)
#from PyQt5.QtGui import QIcon
#from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
import argparse
from pygame import mixer
import subprocess
import time
import socket

HEADER_SIZE=4
NAME_SIZE=20
IMAGE_QUALITY=30

def pil2cv(image):
    ''' PIL -> OpenCV '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  #mono
        pass
    elif new_image.shape[2] == 3:  #color
        #new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        print("bgr")
    elif new_image.shape[2] == 4:  #aer
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        print("A")
    return new_image

class TakePicture:
      def __init__(self,x0,y0,x1,y1):
          self.x0=x0
          self.x1=x1
          self.y0=y0
          self.y1=y1
          self.width=x1-x0
          self.height=y1-y0
          print(self.width,self.height)
      def get_frame(self):
          #print(self.x0,self.y0,self.x1,self.y1)
          #self.img=ImageGrab.grab(bbox=(self.x0,self.y0,self.x1,self.y1))
          #frame=pil2cv(self.img)
          #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
          img=ImageGrab.grab(bbox=(self.x0,self.y0,self.x1,self.y1))
          print(img)
          frame = np.array(img, dtype=np.uint8)
          print("shape",frame.shape)
          return (1,frame)

class MainWindow(QMainWindow):
    #repeatTime=500 #ms
    v_width=640
    v_height=480

    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent)
        
        self.repeatTime =args.itimer #ms Update Time
        self.count=1
        

        self.cap=cv2.VideoCapture(0)
        if self.cap.isOpened() is False:
            raise("IO Error")
        self.view = QGraphicsView()

        size = (self.v_width, self.v_height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # need otherwise select YUV and become slow
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH ,self.v_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.v_height)
        self.setGeometry(0, 0, 840, 400)
        self.setWindowTitle('RockX Face Recog')
        
        #window setup
        self.widget = QWidget()
        self.view = QGraphicsView()

        self.scene = QGraphicsScene()
        #self.scene.setSceneRect(0, 0, 800, 480)
        self.srect = self.view.rect()

        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        self.view.setScene(self.scene)
        #ぴったり合わせるおまじない
        #self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        #レイアウトの作成
        self.main = QGridLayout()
        #label & buttonの表示
        self.label = QLabel("Regist Name")
        self.entry = QLineEdit()
        self.regist = QPushButton('RegMode')
        self.reg_ok = QPushButton('Reg OK')
        self.eval   = QPushButton("Evaluate")
        self.result = QLineEdit()
        self.check=QCheckBox('ini')


        #signalの設定
        self.main.addWidget(self.label,   0,1,1,1)
        self.main.addWidget(self.entry,   1,1,1,1)
        self.main.addWidget(self.regist,  2,1,1,1)
        self.main.addWidget(self.reg_ok,  3,1,1,1)
        self.main.addWidget(self.check,   4,1,1,1)
        self.main.addWidget(self.eval,    5,1,1,1)
        self.main.addWidget(self.result,  6,1,1,1)
        self.main.addWidget(self.view, 0, 0, 20,1)

        self.widget.setLayout(self.main)
        self.setCentralWidget(self.widget)

        self.regist.clicked.connect(self.buttonRegMode)
        self.reg_ok.clicked.connect(self.buttonOK)
        self.eval.clicked.connect(self.EvalMode)
        self.check.stateChanged.connect(self.CheckState)

        self.mode=0
        self.reg_on=0
        self.x0=0
        self.x1=0
        self.y0=0
        self.y1=0
        self.faceinit=0
        self.check_state=0

        self.mes='                   '
        self.face_db,reco_keys=face_reco.rock_init()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #s.bind(('192.168.2.115', 50007))
        s.bind((args.ip, 50007))
        s.listen(1)
        self.soc, self.addr = s.accept()
        print("Recv Address",self.addr)

        self.set()


        #update timer
        timer = QTimer(self.view)
        timer.timeout.connect(self.set)
        timer.start(self.repeatTime)

    def InitDB(self):
        #self.browser.reload()
        self.browser.load(QUrl(self.initurl)) 
        #subprocess.run(['sh', 'removeFace.sh'])
        print("******reload******")

    def buttonRegMode(self):
        self.mode=1
        print("RegMode")

    def buttonOK(self):
        self.reg_on=1
        fname=self.entry.text()
        print(fname)
        print("reg OK")

    def EvalMode(self):
        self.mode=0

    def eventFilter(self, source, event):
        offset=40
        if event.type() == QtCore.QEvent.MouseMove:
            if event.buttons() == QtCore.Qt.NoButton:
                print("Simple mouse motion",event.x(),event.y())
            elif event.buttons() == QtCore.Qt.LeftButton:
                print("Left click drag",event.x(),event.y())
                self.x1=event.x()-offset
                self.y1=event.y()
            elif event.buttons() == QtCore.Qt.RightButton:
                print("Right click drag",event.x(),event.y())
        elif event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton:
                print("Press!",event.x(),event.y())
                self.x0=event.x()-offset
                self.x1=self.x0
                self.y0=event.y()
                self.y1=self.y0
        return super(MainWindow, self).eventFilter(source, event)

    def CheckState(self,state):
        if state==2:
           self.faceinit=1
           self.check_state=1
        else:
           self.faceinit=0
           self.check_state=0

    def loadFinishedHandler(self):
        print(": load finished")

    def set(self):  # Update Function
        
        self.mes="{:20}".format("Loop")[:NAME_SIZE]
        #camera capture
        ret,frame=self.cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
     
        print("Count ",self.count)
        self.count=self.count+1
        
        if ret:
           ret1,names,boxes,scores=face_reco.rock(frame)
           if ret1 == -1:
              face_reco.rock_init()
        
        if  ret and self.mode == 1:
            self.repeatTime=20
            cv2.rectangle(frame,(self.x0,self.y0),(self.x1,self.y1),(0,0,255),thickness=2)
            if self.reg_on == 1 and self.x0 != self.x1 and self.y0 != self.y1 and self.faceinit==0:
                cut_frame=frame[self.y0:self.y1,self.x0:self.x1]
                fname=self.entry.text()
                print("FNAME ",fname)
                cv2.imwrite("./image/"+fname+".jpg",cut_frame)
                self.reg_on =0
                self.mode=0
                print(self.face_db)
                face_reco.import_face(self.face_db)
                self.repeatTime=args.itimer
                if self.check_state==1:
                   self.faceinit=1
                self.y0=0
                self.y1=0
                self.x0=0
                self.x1=0
            elif self.reg_on == 1 and self.x0 != self.x1 and self.y0 != self.y1 and self.check_state==1:
                subprocess.run(['sh', 'removeFace.sh'])
                self.faceinit=0
                
    
        #print(names)
        
        if ret and self.mode == 0:  #evaluation
            msg=None
            #self.result.setText(msg)
            f=1.0
            for box,name,score in zip(boxes,names,scores):
                if name is not None:
                   msg=name+"("+str(round(score,2))+")"
                   self.result.setText(msg)
                   print(name)
                   self.mes="{:20}".format("Recog:"+name+"("+str(round(score,2))+")")[:NAME_SIZE]
                   cv2.rectangle(frame,(box.left,box.top),(box.right,box.bottom),(0,255,0),thickness=1)
                   if name == args.t_name:
                      if args.sound == 1 :
                         mixer.music.play(1)
                else:
                   msg="No Recognize"
                   self.result.setText(msg)
                   print("no recognize")
                   self.mes="{:20}".format("NoReco:")[:NAME_SIZE]
                f=f+1.0
            loop_start_time = time.time()

            send_img=cv2.resize(frame,(640,480))#
            (status, encoded_img) = cv2.imencode('.jpg', send_img, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])
            packet_body = encoded_img.tostring()
            #packet_name = 'TakiguchiT'.encode()
            print(self.mes)
            packet_name = self.mes.encode()
            packet_header = len(packet_body).to_bytes(HEADER_SIZE, 'big') 
            packet = packet_header + packet_name+packet_body
            #packet = packet_header + packet_body
            print("PAK",len(packet_body),packet_header)
            self.soc.sendall(packet)
            FPS=10
            time.sleep(max(0, 1 / FPS - (time.time() - loop_start_time)))
        
        height, width, dim = frame.shape
        bytesPerLine = dim * width
        self.image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.item = QGraphicsPixmapItem(QPixmap.fromImage(self.image))
        self.scene.addItem(self.item)
        self.view.setScene(self.scene)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RockX Face Recognition Demo")
    parser.add_argument('-s','--sound',help="sound",type=int,default=0)
    parser.add_argument('-n','--t_name',help='target name',type=str,default='taki')
    parser.add_argument('-a','--apmode',help='wifi ap mode',type=int,default=0)
    parser.add_argument('-w','--wifi',help='wifi cam',type=int,default=1)
    parser.add_argument('-t','--itimer',help='interval time',type=int,default=520)
    parser.add_argument('-i','--ip',help='interval time',default="192.168.2.115")
    args=parser.parse_args()
    mixer.init()
    mixer.music.load("button06.mp3")


    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    app.exec_()

