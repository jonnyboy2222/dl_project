import socket
import json
import cv2
import threading
import time
import numpy as np
import sys
import queue
import struct
import atexit

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

from threading import Lock

from distance import estimate_stopline_distance

import base64
import pandas as pd

# 서버 IP 및 포트 정보
LANE_SERVER_IP = "192.168.0.252"
TCP_LANE_PORT = 12345
UDP_LANE_PORT = 54321

OBJ_SERVER_IP = "192.168.0.102"
TCP_OBJ_PORT = 12346
UDP_OBJ_PORT = 54322

udp_video_queue = queue.Queue()

lane_tcp_queue = queue.Queue()
obj_tcp_queue = queue.Queue()

lane_result_queue = queue.Queue()
obj_result_queue = queue.Queue()

orig_frame = {}
lane_mask = {}
obj_mask = {}

HEADER_LENGTH = 4
UUID_LENGTH = 4

original_latency_check = {}
lane_latency_check = {}
obj_latency_check = {}


from_class = uic.loadUiType("/home/lee/dev_ws/projects/DL_project/gui/client_video.ui")[0]

class UdpSender():
    def __init__(self):
        self.udp_lane = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.uuid_counter = 0

    def send_frame(self):
        global original_latency_check

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

                if not ret:
                    continue
                
                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if not ret:
                    continue

                try:
                    self.udp_lane.sendto(uuid_msg + b'||' + buffer.tobytes(), (LANE_SERVER_IP, UDP_LANE_PORT))
                    self.udp_obj.sendto(uuid_msg + b'||' + buffer.tobytes(), (OBJ_SERVER_IP, UDP_OBJ_PORT))

                    udp_video_queue.put((self.uuid_counter, frame.copy()))


                    # latency_check
                    original_latency_check[self.uuid_counter] = time.time()

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

    def decode_mask_png_base64(encoded: str) -> np.ndarray:
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return mask  # dtype=np.uint8, shape=(H, W)

    def receive_data(self):
        while True:
            try:
                # 먼저 4바이트 헤더 읽기
                header = self.tcp_lane.recv(HEADER_LENGTH)
                if len(header) < HEADER_LENGTH:
                    raise ValueError("Incomplete header")

                json_len = struct.unpack('>I', header)[0]

                uuid_raw = self.tcp_lane.recv(UUID_LENGTH)
                if len(uuid_raw) < UUID_LENGTH:
                    raise ValueError("Incomplete uuid")

                uuid = struct.unpack('>I', uuid_raw)[0]

                # 정확히 그 길이만큼 받기
                buffer = b''
                while len(buffer) < json_len:
                    chunk = self.tcp_lane.recv(json_len - len(buffer))
                    if not chunk:
                        raise ConnectionError("Socket closed unexpectedly")
                    buffer += chunk

                result = json.loads(buffer.decode('utf-8'))

                if "pred_mask" not in result:
                    print(f"[WARN] pred_mask not in result for UUID {uuid}")
                    continue

                # base64 → ndarray 변환
                pred_mask = self.decode_mask_png_base64(result["pred_mask"])

                lane_tcp_queue.put(uuid, pred_mask)
                
                
                # return json_data
            
            except Exception as e:
                print(f"[TCP LANE RECEIVE ERROR] {e}")
                return None
            
    def close(self):
        if self.tcp_lane is not None:
            self.tcp_lane.close()
            self.tcp_lane = None

class LaneResultProcessor():
    def __init__(self):
        pass

    def process_result(self):
        try:
            while not lane_tcp_queue.empty():
                uuid, pred_mask = lane_tcp_queue.get()
                
                if pred_mask is None:
                    continue

                h, w = pred_mask.shape

                left_zone = pred_mask[int(h * 0.5):, int(w * 0.2):int(w * 0.4)]
                right_zone = pred_mask[int(h * 0.5):, int(w * 0.6):int(w * 0.8)]

                can_change_left = np.count_nonzero(left_zone == 2) > 40
                can_change_right = np.count_nonzero(right_zone == 2) > 40

                stop_line = np.count_nonzero(pred_mask == 4) > 50
                crosswalk = np.count_nonzero(pred_mask == 5) > 50

                msg = [0, 0, 0, 0, 0]
                if can_change_left:
                    msg[0] = 1
                if can_change_right:
                    msg[1] = 1
                if not (can_change_left or can_change_right):
                    msg[2] = 1
                if stop_line:
                    msg[3] = 1
                if crosswalk:
                    msg[4] = 1

                lane_result_queue.put((uuid, pred_mask, msg))

                # latency_check
                lane_latency_check[uuid] = time.time()

                # return uuid, pred_mask


        except Exception as e:
            print(f"[LANE RESULT PROCESS ERROR] {e}")
            return None



class TcpObjReceiver():
    def __init__(self):
        self.tcp_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_obj.connect((OBJ_SERVER_IP, TCP_OBJ_PORT))
        self.tcp_obj.settimeout(1.0)

    def receive_data(self):
        while True:
            try:
                # 먼저 4바이트 헤더 읽기
                header = self.tcp_obj.recv(HEADER_LENGTH)
                if len(header) < 4:
                    raise ValueError("Incomplete header")

                json_len = struct.unpack('>I', header)[0]

                uuid_raw = self.tcp_obj.recv(UUID_LENGTH)
                if len(uuid_raw) < UUID_LENGTH:
                    raise ValueError("Incomplete uuid")
                
                uuid = struct.unpack('>I', uuid_raw)[0]

                # 정확히 그 길이만큼 받기
                buffer = b''
                while len(buffer) < json_len:
                    chunk = self.tcp_obj.recv(json_len - len(buffer))
                    if not chunk:
                        raise ConnectionError("Socket closed unexpectedly")
                    buffer += chunk

                json_data = json.loads(buffer.decode('utf-8'))

                obj_tcp_queue.put(uuid, json_data)
                
                # return json_data
            
            except Exception as e:
                print(f"[TCP OBJ RECEIVE ERROR] {e}")
                return None
            
    def close(self):
        if self.tcp_obj is not None:
            self.tcp_obj.close()
            self.tcp_obj = None

class ObjectResultProcessor():
    def __init__(self):
        pass

    def process_result(self):
        try:
            # 큐에서 최신 결과 추출
            while not obj_tcp_queue.empty():
                uuid, obj_data = obj_tcp_queue.get()

                if obj_data is None:
                    continue

                # 프레임과 동일한 크기의 빈 overlay 생성
                mask = np.zeros((256, 512), dtype=np.uint8)

                # detection 결과를 mask에 그림
                if 'bbox' not in obj_data:
                    continue
                
                x1, y1, x2, y2 = obj_data['bbox']
                class_id, class_name = obj_data.get('class_id', 'class_name')

                label = f"{class_name}"
                color = (0, 255, 0)

                cv2.rectangle(mask, (x1, y1), (x2, y2), color, 2)
                cv2.putText(mask, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                obj_result_queue.put((uuid, mask, class_id))

                # latency_check
                obj_latency_check[uuid] = time.time()

                # return mask 

        except Exception as e:
            print(f"[OBJ RESULT PROCESS ERROR] {e}")
            return None

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("COVA II")

        # self.timer = QTimer()
        # self.timer.timeout.connect(self.update_video_gui)
        # self.timer.start(50)  # ~30fps

        # 신호등과 정지선
        self.state = True # Moving
        self.prev_state = True

        self.udp_sender = UdpSender()
        self.tcp_lane_receiver = TcpLaneReceiver()
        self.tcp_obj_receiver = TcpObjReceiver()
        self.lane_result_processor = LaneResultProcessor()
        self.obj_result_processor = ObjectResultProcessor()



        threading.Thread(target=self.udp_sender.send_frame, daemon=True).start()
        threading.Thread(target=self.tcp_lane_receiver.receive_data, daemon=True).start()
        threading.Thread(target=self.tcp_obj_receiver.receive_data, daemon=True).start()
        threading.Thread(target=self.lane_result_processor.process_result, daemon=True).start()
        threading.Thread(target=self.obj_result_processor.process_result, daemon=True).start()

    
    # def update_video_gui(self):
    #     global orig_frame, lane_mask, obj_mask

    #     if not udp_video_queue.empty():
    #         frame = udp_video_queue.get() # uuid, frame

    #         orig_frame[frame[0]] = frame[1]
        

    #     if not lane_result_queue.empty():
    #         lane_result = lane_result_queue.get() # uuid, pred_mask, msg

    #         lane_mask[lane_result[0]] = lane_result[1]
    #         msg_one_hot_vector = lane_result[2]

    #     if not obj_result_queue.empty():
    #         obj_result = obj_tcp_queue.get() # uuid, overlay, cls_id

    #         obj_mask[obj_result[0]] = obj_result[1]
    #         cls_id = obj_result[2]




    #     if frame is not None:
            

    #         angle = lane_result.get("steering_angle", 0.0)
    #         self.label_msg_angle.setText(f"Angle: {angle:.2f}")

    #         real_distance = lane_result.get("real_distance", 0.0)
    #         self.state = True        
    #         if real_distance < 2.0 and obj_result.get("class_name") == "vehicle_stop":
    #             self.label_msg_alert.setText("STOP")
    #             self.state = False

    #         if self.state==True and self.prev_state==False:
    #             self.label_msg_alert.setText("GO")
    #             self.prev_state = self.state

    #         # 차선 변경 가능 유무 메세지
    #         self.label_msg_lane.setText("차선 변경이 가능합니다")



    #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgb.shape
    #     bytes_per_line = ch * w
    #     img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    #     pixmap = QPixmap.fromImage(img)
    #     self.label_video_lane.setPixmap(pixmap.scaled(
    #         self.label_video_lane.width(), self.label_video_lane.height(), Qt.AspectRatioMode.KeepAspectRatio))

    # def closeEvent(self, event):
    #     self.udp_sender.close()
    #     self.tcp_lane_receiver.close()
    #     self.tcp_obj_receiver.close()
    #     event.accept() # 창 닫기 허용

# Main
if __name__ == "__main__":
    df_orig = pd.DataFrame.from_dict(original_latency_check, orient='index')
    df_lane = pd.DataFrame.from_dict(lane_latency_check, orient='index')
    df_obj  = pd.DataFrame.from_dict(obj_latency_check, orient='index')

    df_merged = pd.concat([df_orig, df_lane, df_obj], axis=1)

    df_merged.to_csv("latency_summary.csv", index_label="uuid")



    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    sys.exit(app.exec())
