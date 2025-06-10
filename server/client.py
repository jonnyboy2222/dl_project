import socket
import json
import cv2
import threading
import time
import numpy as np
import sys
from queue import Queue
import struct

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

# 서버 IP 및 포트 정보
LANE_SERVER_IP = "192.168.0.252"
TCP_LANE_PORT = 12345
UDP_LANE_PORT = 54321

OBJ_SERVER_IP = "192.168.2.102"
TCP_OBJ_PORT = 12346
UDP_OBJ_PORT = 54322

from_class = uic.loadUiType("/home/lee/dev_ws/projects/DL_project/gui/client_video.ui")[0]

class TcpLaneClientThread(QThread):
    msg_lane = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            try:
                client.connect((LANE_SERVER_IP, TCP_LANE_PORT))
                client.settimeout(1.0)

                buffer = b""
                expected_size = 10

                while self.running:
                    try:
                        data = client.recv(1024)
                        if not data:
                            continue

                        buffer += data

                        while len(buffer) >= expected_size:
                            uuid, angle, n_points = struct.unpack('>I f H', buffer)
                            header_str = uuid.decode('ascii')
                            angle_float = float(angle)
                            n_points_int = int(n_points)

                            parsed_msg = f"[{header_str} Angle:{angle_float:.2f}, N_Points:{n_points_int}]"
                            print(parsed_msg)
                            self.msg_lane.emit(parsed_msg)
                            
                    except socket.timeout:
                        continue
            except Exception as e:
                print(f"[TCP LANE ERROR] {e}")

    def stop(self):
        self.running = False
        self.wait()


class TcpObjClientThread(QThread):
    msg_obj = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            try:
                client.connect((OBJ_SERVER_IP, TCP_OBJ_PORT))
                client.settimeout(1.0)

                buffer = b""
                expected_size = 16

                while self.running:
                    try:
                        data = client.recv(1024)
                        if not data:
                            continue

                        buffer += data

                        while len(buffer) >= expected_size:
                            chunk = buffer[:expected_size]
                            buffer = buffer[expected_size:]

                            header, value, flag = struct.unpack('>4s f B', chunk)
                            header_str = header.decode('ascii')

                            parsed_msg = f"[{header_str} Value:{value:.2f}, Flag:{flag}]"
                            print(parsed_msg)
                            self.msg_obj.emit(parsed_msg)

                    except socket.timeout:
                        continue
            except Exception as e:
                print(f"[TCP OBJ ERROR] {e}")

    def stop(self):
        self.running = False
        self.wait()


class UdpSenderThread(QThread):
    frame_from_lane = pyqtSignal(np.ndarray)
    frame_from_obj = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True
        self.uuid_counter = 0

    def run(self):
        cap = cv2.VideoCapture('/home/lee/dev_ws/projects/DL_project/lane_detect/UFLDv2_like/video/example.mp4')
        if not cap.isOpened():
            print("[UDP] Webcam open failed")
            return

        udp_lane = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ret:
                continue

            self.uuid_counter += 1
            uuid_msg = self.uuid_counter.to_bytes(4, byteorder='big')
            print(self.uuid_counter,uuid_msg)

            try:
                udp_lane.sendto(uuid_msg + b'||' + buffer.tobytes(), (LANE_SERVER_IP, UDP_LANE_PORT))
                udp_obj.sendto(uuid_msg + b'||' + buffer.tobytes(), (OBJ_SERVER_IP, UDP_OBJ_PORT))
            except Exception as e:
                print(f"[UDP SEND ERROR] {e}")

            # 단순히 현재 프레임 표시용
            self.frame_from_lane.emit(frame)
            self.frame_from_obj.emit(frame)
            self.msleep(33)

        cap.release()

    def stop(self):
        self.running = False


class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("COVA II")

        self.tcp_lane_client = TcpLaneClientThread()
        self.tcp_obj_client = TcpObjClientThread()
        self.udp_sender = UdpSenderThread()

        self.tcp_lane_client.msg_lane.connect(self.update_lane_msg)
        self.tcp_obj_client.msg_obj.connect(self.update_obj_msg)
        self.udp_sender.frame_from_lane.connect(self.update_video_lane)
        # self.udp_sender.frame_from_obj.connect(self.update_video_obj)

        self.tcp_lane_client.start()
        self.tcp_obj_client.start()
        self.udp_sender.start()

    def update_lane_msg(self, msg):
        self.label_msg_lane.setText("Lane: " + msg)

    def update_obj_msg(self, msg):
        self.label_msg_obj.setText("Object: " + msg)

    def update_video_lane(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.label_video_lane.setPixmap(pixmap.scaled(self.label_video_lane.width(), self.label_video_lane.height(), Qt.AspectRatioMode.KeepAspectRatio))

    # def update_video_obj(self, frame):
    #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgb.shape
    #     bytes_per_line = ch * w
    #     img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    #     pixmap = QPixmap.fromImage(img)
    #     self.label_video_obj.setPixmap(pixmap.scaled(self.label_video_obj.width(), self.label_video_obj.height(), Qt.AspectRatioMode.KeepAspectRatio))

# Main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    sys.exit(app.exec())


# class TcpClientThread(QThread):
#     msg_lane = pyqtSignal(str)
#     msg_obj = pyqtSignal(str)

#     def __init__(self):
#         super().__init__()
#         self.running = True

#     def run(self):
#         while self.running:
#             threading.Thread(target=self.tcp_lane_listener).start()
#             threading.Thread(target=self.tcp_obj_listener).start()
#             break

#     def tcp_lane_listener(self):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
#             try:
#                 client.connect((LANE_SERVER_IP, TCP_LANE_PORT))
#                 client.settimeout(1.0)
#                 while self.running:
#                     try:
#                         data = client.recv(1024)
#                         if data:
#                             msg = json.loads(data.decode('utf-8')) # byte buffer로 수정
#                             self.msg_lane.emit(str(msg))
#                     except socket.timeout:
#                         continue
#             except Exception as e:
#                 print(f"[TCP LANE ERROR] {e}")

#     def tcp_obj_listener(self):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
#             try:
#                 client.connect((OBJ_SERVER_IP, TCP_OBJ_PORT))
#                 client.settimeout(1.0)
#                 while self.running:
#                     try:
#                         data = client.recv(1024)
#                         if data:
#                             msg = json.loads(data.decode('utf-8')) # byte buffer로 수정
#                             self.msg_obj.emit(str(msg))
#                     except socket.timeout:
#                         continue
#             except Exception as e:
#                 print(f"[TCP OBJ ERROR] {e}")

#     def stop(self):
#         self.running = False