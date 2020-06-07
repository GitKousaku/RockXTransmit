import os
import sys
import time
import sqlite3
import argparse
import numpy as np

from rockx import RockX
import cv2
from time import sleep



class FaceDB:

    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()
        if not self._is_face_table_exist():
            self.cursor.execute("create table FACE (NAME text, VERSION int, FEATURE blob, ALIGN_IMAGE blob)")

    def load_face(self):
        all_face = dict()
        c = self.cursor.execute("select * from FACE")
        for row in c:
            name = row[0]
            version = row[1]
            feature = np.frombuffer(row[2], dtype='float32')
            align_img = np.frombuffer(row[3], dtype='uint8')
            align_img = align_img.reshape((112, 112, 3))
            all_face[name] = {
                'feature': RockX.FaceFeature(version=version, len=feature.size, feature=feature),
                'image': align_img
            }
        return all_face

    def insert_face(self, name, feature, align_img):
        self.cursor.execute("INSERT INTO FACE (NAME, VERSION, FEATURE, ALIGN_IMAGE) VALUES (?, ?, ?, ?)",
                            (name, feature.version, feature.feature.tobytes(), align_img.tobytes()))
        self.conn.commit()

    def _get_tables(self):
        cursor = self.cursor
        cursor.execute("select name from sqlite_master where type='table' order by name")
        tables = cursor.fetchall()
        return tables

    def _is_face_table_exist(self):
        tables = self._get_tables()
        for table in tables:
            if 'FACE' in table:
                return True
        return False


def get_max_face(results):
    max_area = 0
    max_face = None
    for result in results:
        area = (result.box.bottom - result.box.top) * (result.box.right * result.box.left)
        if area > max_area:
            max_face = result
    return max_face


def get_face_feature(image_path):
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    ret, results = face_det_handle.rockx_face_detect(img, img_w, img_h, RockX.ROCKX_PIXEL_FORMAT_BGR888)
    if ret != RockX.ROCKX_RET_SUCCESS:
        return None, None
    max_face = get_max_face(results)
    if max_face is None:
        return None, None
    ret, align_img = face_landmark5_handle.rockx_face_align(img, img_w, img_h,
                                                            RockX.ROCKX_PIXEL_FORMAT_BGR888,
                                                            max_face.box, None)
    if ret != RockX.ROCKX_RET_SUCCESS:
        return None, None
    if align_img is not None:
        ret, face_feature = face_recog_handle.rockx_face_recognize(align_img)
        if ret == RockX.ROCKX_RET_SUCCESS:
            return face_feature, align_img
    return None, None


def get_all_image(image_path):
    img_files = dict()
    g = os.walk(image_path)

    for path, dir_list, file_list in g:
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if not os.path.isdir(file_path):
                img_files[os.path.splitext(file_name)[0]] = file_path
    return img_files


def import_face(face_db, images_dir="./image"):
    global face_library
    image_files = get_all_image(images_dir)
    image_name_list = list(image_files.keys())
    for name, image_path in image_files.items():
        print(name,image_path)
        feature, align_img = get_face_feature(image_path)
        if feature is not None:
            face_db.insert_face(name, feature, align_img)
            #print('[%d/%d] success import %s ' % (image_name_list.index(name)+1, len(image_name_list), image_path))
        else:
            hhhh=1
            #print('[%d/%d] fail import %s' % (image_name_list.index(name)+1, len(image_name_list), image_path))
    face_library = face_db.load_face()

def search_face(face_library, cur_feature):
    global face_recog_handle
    min_similarity = 10.0
    target_name = None
    target_face = None
    for name, face in face_library.items():
        feature = face['feature']
        ret, similarity = face_recog_handle.rockx_face_similarity(cur_feature, feature)
        if similarity < min_similarity:
            target_name = name
            min_similarity = similarity
            target_face = face
    if min_similarity < 1.0:
        return target_name, min_similarity, target_face
    return None, -1, None

def rock_init():
    global face_recog_handle
    global face_db
    global face_library
    global face_det_handle
    global face_track_handle
    global face_landmark5_handle

    camera_num=0
    db_file="face.db"
    target_device=None
    image_dir="./image"

    print("rock init")
    face_det_handle = RockX(RockX.ROCKX_MODULE_FACE_DETECTION, target_device)
    face_landmark5_handle = RockX(RockX.ROCKX_MODULE_FACE_LANDMARK_5, target_device)
    face_recog_handle = RockX(RockX.ROCKX_MODULE_FACE_RECOGNIZE, target_device)
    face_track_handle = RockX(RockX.ROCKX_MODULE_OBJECT_TRACK, target_device)

    face_db = FaceDB(db_file)
    #import_face(face_db, image_dir)
    # load face from database
    face_library = face_db.load_face()
    print("load %d face" % len(face_library))
    print(face_library.keys())
    return face_db,face_library.keys()

def rock(frame):

    global face_recog_handle
    global face_det_handle
    global face_db
    global face_library
    global face_track_handle
    global face_landmark5_handle

    #
    in_img_h, in_img_w = frame.shape[:2]
    retx, results = face_det_handle.rockx_face_detect(frame, in_img_w, in_img_h, RockX.ROCKX_PIXEL_FORMAT_BGR888)
    print("face_detect ret",retx)
    ret, results = face_track_handle.rockx_object_track(in_img_w, in_img_h, 3, results)
    

    index = 0
    boxes=[]
    names=[]
    scores=[]
    ret1=0
    for result in results:
            target_name=""
            diff=-1

            # face align
            
            ret, align_img = face_landmark5_handle.rockx_face_align(frame, in_img_w, in_img_h,
                                                                     RockX.ROCKX_PIXEL_FORMAT_BGR888,
                                                                    result.box, None)
            print("landmark",ret);

            # get face feature
            if ret == RockX.ROCKX_RET_SUCCESS and align_img is not None:
                ret, face_feature = face_recog_handle.rockx_face_recognize(align_img)
                print("faceeco",ret)

            # search face
            if ret == RockX.ROCKX_RET_SUCCESS and face_feature is not None:
                target_name, diff, target_face = search_face(face_library, face_feature)
                print("serach",target_name)
                #print("target_name=%s diff=%s", target_name, str(diff))
            
            boxes.append(result.box)
            names.append(target_name)
            scores.append(diff)


    return retx,names,boxes,scores



if __name__ == '__main__':
    global face_recog_handle
    
    '''
    parser = argparse.ArgumentParser(description="RockX Face Recognition Demo")
    parser.add_argument('-c', '--camera', help="camera index", type=int, default=0)
    parser.add_argument('-b', '--db_file', help="face database path", required=True)
    parser.add_argument('-i', '--image_dir', help="import image dir")
    parser.add_argument('-d', '--device', help="target device id", type=str)
    args = parser.parse_args()

    print("camera=", args.camera)
    print("db_file=", args.db_file)
    print("image_dir=", args.image_dir)
    print("target device=",args.device)

    
    '''
    camera_num=0
    db_file="face.db"
    #target_device=None
    #image_dir=None
    target_device=""
    image_dir=""
    



    face_det_handle = RockX(RockX.ROCKX_MODULE_FACE_DETECTION, target_device)
    print("2")
    face_landmark5_handle = RockX(RockX.ROCKX_MODULE_FACE_LANDMARK_5, target_device)
    face_recog_handle = RockX(RockX.ROCKX_MODULE_FACE_RECOGNIZE, target_device)
    face_track_handle = RockX(RockX.ROCKX_MODULE_OBJECT_TRACK, target_device)
    face_db = FaceDB(db_file)

    '''
    camera_num=0
    print("1")
    print(args.device)
    face_det_handle = RockX(RockX.ROCKX_MODULE_FACE_DETECTION, target_device=args.device)
    target_device=args.device
    print(target_device)
    print("2")
    face_landmark5_handle = RockX(RockX.ROCKX_MODULE_FACE_LANDMARK_5, target_device=args.device)
    face_recog_handle = RockX(RockX.ROCKX_MODULE_FACE_RECOGNIZE, target_device=args.device)
    face_track_handle = RockX(RockX.ROCKX_MODULE_OBJECT_TRACK, target_device=args.device)
    face_db = FaceDB(args.db_file)
    '''


    # load face from database
    face_library = face_db.load_face()
    print("load %d face" % len(face_library))
    print(face_library)

    cap = cv2.VideoCapture(camera_num)
    cap.set(3, 1280)
    cap.set(4, 720)


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print("KKKKKK")
        #print(ret)
        show_frame = frame.copy()

        in_img_h, in_img_w = frame.shape[:2]
        ret, results = face_det_handle.rockx_face_detect(frame, in_img_w, in_img_h, RockX.ROCKX_PIXEL_FORMAT_BGR888)

        ret, results = face_track_handle.rockx_object_track(in_img_w, in_img_h, 3, results)
        target_name=""

        index = 0
        for result in results:
            # face align
            ret, align_img = face_landmark5_handle.rockx_face_align(frame, in_img_w, in_img_h,
                                                                     RockX.ROCKX_PIXEL_FORMAT_BGR888,
                                                                     result.box, None)

            # get face feature
            if ret == RockX.ROCKX_RET_SUCCESS and align_img is not None:
                ret, face_feature = face_recog_handle.rockx_face_recognize(align_img)

            # search face
            if ret == RockX.ROCKX_RET_SUCCESS and face_feature is not None:
                target_name, diff, target_face = search_face(face_library, face_feature)
                #print("target_name=%s diff=%s", target_name, str(diff))

            # draw
            cv2.rectangle(show_frame, (result.box.left, result.box.top), (result.box.right, result.box.bottom), (0, 255, 0), 2)
            if align_img is not None and index < 3:
                show_frame[(10+112*index):(10+112*index+112), 10:(112+10)] = align_img

            show_str = str(result.id)
            if target_name is not None:
                show_str += str.format(" - {name}", name=target_name)

            if target_face is not None and 'image' in target_face.keys() and target_face['image'] is not None and index < 3:
                show_frame[(10+112*index):(10+112*index+112), 132:(112+132)] = target_face['image']

            cv2.putText(show_frame, show_str, (result.box.left, result.box.top-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            index += 1
        # Display the resulting frame
        cv2.imshow('RockX Face Recog - ' + str(target_device), show_frame)
        cv2.moveWindow('RockX Face Recog - ' + str(target_device), 300, 100)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    face_det_handle.release()

