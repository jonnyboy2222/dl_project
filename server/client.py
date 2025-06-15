import socket
import json
import cv2
import threading
import time
import numpy as np
import sys
from queue import Queue
import struct
import atexit

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

from threading import Lock

# 서버 IP 및 포트 정보
LANE_SERVER_IP = "192.168.0.252"
TCP_LANE_PORT = 12345
UDP_LANE_PORT = 54321

OBJ_SERVER_IP = "192.168.0.102"
TCP_OBJ_PORT = 12346
UDP_OBJ_PORT = 54322

latest_frame = None
latest_lane_result = None
frame_lock = Lock()
json_lock = Lock()


from_class = uic.loadUiType("/home/lee/dev_ws/projects/DL_project/gui/client_video.ui")[0]

class UdpSender():
    def __init__(self):
        self.udp_lane = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.uuid_counter = 0

    def send_frame(self):
        global latest_frame, frame_lock

        self.cap = cv2.VideoCapture(0)

        try:
            if not self.cap.isOpened():
                print("[UDP] Webcam open failed")
                return

            while True:
                self.uuid_counter += 1
                uuid_msg = self.uuid_counter.to_bytes(4, byteorder='big')

                ret, frame = self.cap.read()

                frame = cv2.resize(frame, (640, 480))

                with frame_lock:
                    latest_frame = frame.copy()

                if not ret:
                    continue
                
                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if not ret:
                    continue

                try:
                    self.udp_lane.sendto(uuid_msg + b'||' + buffer.tobytes(), (LANE_SERVER_IP, UDP_LANE_PORT))
                    self.udp_obj.sendto(uuid_msg + b'||' + buffer.tobytes(), (OBJ_SERVER_IP, UDP_OBJ_PORT))
                except Exception as e:
                    print(f"[UDP SEND ERROR] {e}")
        finally:
            self.cap.release()


    def close(self):
        self.udp_lane.close()
        self.udp_obj.close()
        self.cap.release()



class TcpLaneReceiver():
    def __init__(self):
        self.tcp_lane = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_lane.connect((LANE_SERVER_IP, TCP_LANE_PORT))
        self.tcp_lane.settimeout(1.0)

    def receive_data(self):
        global latest_lane_result, json_lock
        while True:
            try:
                # 먼저 4바이트 헤더 읽기
                header = self.tcp_lane.recv(4)
                if len(header) < 4:
                    raise ValueError("Incomplete header")

                json_len = struct.unpack('>I', header)[0]

                # 정확히 그 길이만큼 받기
                buffer = b''
                while len(buffer) < json_len:
                    chunk = self.tcp_lane.recv(json_len - len(buffer))
                    if not chunk:
                        raise ConnectionError("Socket closed unexpectedly")
                    buffer += chunk

                json_data = json.loads(buffer.decode('utf-8'))

                with json_lock:
                    latest_lane_result = json_data
                
                return json_data
            
            except Exception as e:
                print(f"[TCP LANE RECEIVE ERROR] {e}")
                return None
            
    def close(self):
        if self.tcp_lane is not None:
            self.tcp_lane.close()
            self.tcp_lane = None
            
class TcpObjReceiver():
    def __init__(self):
        self.tcp_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_obj.connect((OBJ_SERVER_IP, TCP_OBJ_PORT))
        self.tcp_obj.settimeout(1.0)

    def receive_data(self):
        while True:
            try:
                # 먼저 4바이트 헤더 읽기
                header = self.tcp_obj.recv(4)
                if len(header) < 4:
                    raise ValueError("Incomplete header")

                json_len = struct.unpack('>I', header)[0]

                # 정확히 그 길이만큼 받기
                buffer = b''
                while len(buffer) < json_len:
                    chunk = self.tcp_obj.recv(json_len - len(buffer))
                    if not chunk:
                        raise ConnectionError("Socket closed unexpectedly")
                    buffer += chunk

                json_data = json.loads(buffer.decode('utf-8'))
                
                return json_data
            
            except Exception as e:
                print(f"[TCP OBJ RECEIVE ERROR] {e}")
                return None
            
    def close(self):
        if self.tcp_obj is not None:
            self.tcp_obj.close()
            self.tcp_obj = None









class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("COVA II")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_gui)
        self.timer.start(33)  # ~30fps

        self.udp_sender = UdpSender()
        self.tcp_lane_receiver = TcpLaneReceiver()
        self.tcp_obj_receiver = TcpObjReceiver()

        threading.Thread(target=self.udp_sender.send_frame, daemon=True).start()
        threading.Thread(target=self.tcp_lane_receiver.receive_data, daemon=True).start()
        threading.Thread(target=self.tcp_obj_receiver.receive_data, daemon=True).start()

    def draw_result_on_frame(self, frame, result_json):
        if not frame.any():
            return frame

        annotated = frame.copy()
        try:
            if "center_line" in result_json:
                pts = result_json["center_line"]
                for i in range(len(pts) - 1):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[i + 1])
                    cv2.line(annotated, pt1, pt2, (0, 255, 255), 2)  # Yellow

            if "lanes" in result_json:
                for lane in result_json["lanes"]:
                    lane_pts = lane["points"]
                    color = (255, 0, 0) if lane["class_name"] == "white_solid" else (0, 255, 0)
                    for i in range(len(lane_pts) - 1):
                        pt1 = tuple(lane_pts[i])
                        pt2 = tuple(lane_pts[i + 1])
                        cv2.line(annotated, pt1, pt2, color, 2)

            angle = result_json.get("steering_angle", 0.0)
            cv2.putText(annotated, f"Steering Angle: {angle:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            print(f"[DRAW ERROR] {e}")

        return annotated
    
    def update_video_gui(self):
        global latest_frame, latest_lane_result, frame_lock, json_lock

        with frame_lock:
            if latest_frame is None:
                return
            frame = latest_frame.copy()

        with json_lock:
            result = latest_lane_result

        if result is not None:
            frame = self.draw_result_on_frame(frame, result)
            angle = result.get("steering_angle", 0.0)
            self.label_msg_lane.setText(f"Angle: {angle:.2f}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.label_video_lane.setPixmap(pixmap.scaled(
            self.label_video_lane.width(), self.label_video_lane.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.udp_sender.close()
        self.tcp_lane_receiver.close()
        self.tcp_obj_receiver.close()
        event.accept() # 창 닫기 허용


# Main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    sys.exit(app.exec())