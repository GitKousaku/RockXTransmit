# coding:utf-8
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import sys
import cv2
import numpy as np
import socket
import configparser
from pygame import mixer

class StreamView(Image):

    def __init__(self, server_ip, server_port, image_width, image_height, view_fps, view_width, view_height, **kwargs):
        print("................")
        super(StreamView, self).__init__(**kwargs)
        print("-----------------------------")
        # �ʐM�p�ݒ�
        self.buff = bytes()
        
        self.PACKET_HEADER_SIZE = 4
        self.PACKET_NAME_SIZE=20  ###19:32
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.SERVER_IP = server_ip
        self.SERVER_PORT = server_port
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        print("********************")

        # �\���ݒ�
        self.allow_stretch = True
        self.VIEW_FPS = view_fps
        self.VIEW_WIDTH = view_width
        self.VIEW_HEIGHT = view_height
        
        #config = configparser.ConfigParser()
        #config_t_name = config.get('other','t_name')
        #config_sound = int(config.get('other','sound'))

        # ��ʍX�V���\�b�h�̌Ăяo���ݒ�
        Clock.schedule_interval(self.update, 1.0 / view_fps)

        # �T�[�o�ɐڑ�
        try:
            self.soc.connect((self.SERVER_IP, self.SERVER_PORT))
        except socket.error as e:
            print('Connection failed.')
            sys.exit(-1)

    def update(self, dt):
        # �T�[�o����̃f�[�^���o�b�t�@�ɒ~��
        data = self.soc.recv(self.IMAGE_HEIGHT * self.IMAGE_WIDTH * 3)
        self.buff += data

        # �ŐV�̃p�P�b�g�̐擪�܂ŃV�[�N
        # �o�b�t�@�ɗ��܂��Ă�p�P�b�g�S�Ă̏����擾
        packet_head = 0
        packets_info = list()
        while True:
            ###if len(self.buff) >= packet_head + self.PACKET_HEADER_SIZE:
            if len(self.buff) >= packet_head + self.PACKET_HEADER_SIZE+self.PACKET_NAME_SIZE:
                binary_size = int.from_bytes(self.buff[packet_head:packet_head + self.PACKET_HEADER_SIZE], 'big')
                name=self.buff[packet_head+ self.PACKET_HEADER_SIZE:packet_head + self.PACKET_HEADER_SIZE+self.PACKET_NAME_SIZE]###
                dec_name=name.decode()
                print("Status:",dec_name)
                dec_name=dec_name[6:dec_name.find("(")]
                #print(dec_name)
                #if dec_name == config_t_name:
                      #if config_sound == 1 :
                         #mixer.music.play(1)
                #if len(self.buff) >= packet_head + self.PACKET_HEADER_SIZE + binary_size:
                if len(self.buff) >= packet_head + self.PACKET_HEADER_SIZE + binary_size+self.PACKET_NAME_SIZE:
                    packets_info.append((packet_head, binary_size))
                    #packet_head += self.PACKET_HEADER_SIZE + binary_size
                    packet_head += self.PACKET_HEADER_SIZE + binary_size+self.PACKET_NAME_SIZE #
                else:
                    break
            else:
                break

        # �o�b�t�@�̒��Ɋ��������p�P�b�g������΁A�摜���X�V
        if len(packets_info) > 0:
            # �ŐV�̊��������p�P�b�g�̏����擾
            packet_head, binary_size = packets_info.pop()
            # �p�P�b�g����摜�̃o�C�i�����擾
            #img_bytes = self.buff[packet_head + self.PACKET_HEADER_SIZE:packet_head + self.PACKET_HEADER_SIZE + binary_size]
            img_bytes = self.buff[packet_head + self.PACKET_HEADER_SIZE+self.PACKET_NAME_SIZE:packet_head + self.PACKET_HEADER_SIZE + binary_size+self.PACKET_NAME_SIZE]
            # �o�b�t�@����s�v�ȃo�C�i�����폜
            #self.buff = self.buff[packet_head + self.PACKET_HEADER_SIZE + binary_size:]
            self.buff = self.buff[packet_head + self.PACKET_HEADER_SIZE + binary_size+self.PACKET_NAME_SIZE:]

            # �摜���o�C�i�����畜��
            img = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img, 1)
            # �摜��\���p�ɉ��H
            img = cv2.flip(img, 0)
            img = cv2.resize(img, (self.VIEW_WIDTH, self.VIEW_HEIGHT))
            # �摜���o�C�i���ɕϊ�
            img = img.tostring()

            # �쐬�����摜���e�N�X�`���ɐݒ�
            img_texture = Texture.create(size=(self.VIEW_WIDTH, self.VIEW_HEIGHT), colorfmt='rgb')
            img_texture.blit_buffer(img, colorfmt='rgb', bufferfmt='ubyte')
            self.texture = img_texture

    def disconnect(self):
        # �T�[�o�Ƃ̐ڑ���ؒf
        self.soc.shutdown(socket.SHUT_RDWR)
        self.soc.close()

class StreamingClientApp(App):

    def __init__(self, view_fps, view_width, view_height, **kwargs):
        super(StreamingClientApp, self).__init__(**kwargs)
        self.VIEW_FPS = view_fps
        self.VIEW_WIDTH = view_width
        self.VIEW_HEIGHT = view_height

    def build(self):
        # �ʐM�p�ݒ���R���t�B�O�t�@�C�����烍�[�h
        config = configparser.ConfigParser()
        config.read('./connection.ini', 'Shift_JIS')
        config_server_ip = config.get('server', 'ip')
        config_server_port = int(config.get('server', 'port'))
        config_header_size = int(config.get('packet', 'header_size'))
        config_image_width = int(config.get('packet', 'image_width'))
        config_image_height = int(config.get('packet', 'image_height'))


        # �E�B���h�E�T�C�Y���r���[�T�C�Y�ɍ��킹��
        
        Window.size = (self.VIEW_WIDTH, self.VIEW_HEIGHT)

        # �X�g���[���r���[�𐶐����A��ʂɐݒ�
        print("===================")
        self.stream_view = StreamView(
            server_ip=config_server_ip,
            server_port=config_server_port,
            image_width=config_image_width,
            image_height=config_image_height,
            view_fps=self.VIEW_FPS,
            view_width=self.VIEW_WIDTH,
            view_height=self.VIEW_HEIGHT
        )
        
        return self.stream_view
        

    def on_stop(self):
        # �T�[�o�Ƃ̐ڑ���ؒf
        self.stream_view.disconnect()

if __name__ == '__main__':
    mixer.init()
    mixer.music.load("button06.mp3")
    #StreamingClientApp(view_fps=30, view_width=800, view_height=600).run()
    StreamingClientApp(view_fps=30, view_width=400, view_height=300).run()