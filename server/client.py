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

HEADER_LENGTH = 4
UUID_LENGTH = 4




from_class = uic.loadUiType("/home/lee/dev_ws/projects/DL_project/gui/client_video.ui")[0]

class UdpSender():
    def __init__(self):
        self.udp_lane = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.uuid_counter = 0

    def send_frame(self):
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

                udp_video_queue.put((self.uuid_counter, frame.copy()))

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
        while True:
            try:
                # 먼저 4바이트 헤더 읽기
                header = self.tcp_lane.recv(HEADER_LENGTH)
                if len(header) < HEADER_LENGTH:
                    raise ValueError("Incomplete header")

                json_len = struct.unpack('>I', header)[0]

                uuid = self.tcp_lane.recv(UUID_LENGTH)
                if len(uuid) < UUID_LENGTH:
                    raise ValueError("Incomplete uuid")

                uuid = struct.unpack('>I', uuid)[0]

                # 정확히 그 길이만큼 받기
                buffer = b''
                while len(buffer) < json_len:
                    chunk = self.tcp_lane.recv(json_len - len(buffer))
                    if not chunk:
                        raise ConnectionError("Socket closed unexpectedly")
                    buffer += chunk

                pred_mask = json.loads(buffer.decode('utf-8'))

                lane_tcp_queue.put((uuid, pred_mask))
                
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
        while True:
            try:
                uuid, lane_data = lane_tcp_queue.get()
                
                if lane_data is None:
                    continue

                pred_mask = np.array(lane_data["pred_mask"], dtype=np.uint8)
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

                # 정확히 그 길이만큼 받기
                buffer = b''
                while len(buffer) < json_len:
                    chunk = self.tcp_obj.recv(json_len - len(buffer))
                    if not chunk:
                        raise ConnectionError("Socket closed unexpectedly")
                    buffer += chunk

                json_data = json.loads(buffer.decode('utf-8'))

                obj_tcp_queue.put(json_data)
                
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

    def process_result(self, frame):
        try:
            # 큐에서 최신 결과 추출
            while not obj_tcp_queue.empty():
                obj_data = obj_tcp_queue.get_nowait()
                # if obj_data is not None:
                #     self.current_detections = obj_data

            # 프레임과 동일한 크기의 빈 overlay 생성
            mask = np.zeros_like(frame, dtype=np.uint8)

            # detection 결과를 mask에 그림
            for det in self.current_detections:
                if 'bbox' not in det:
                    continue
                x1, y1, x2, y2 = det['bbox']
                class_name = det.get('class_name', 'object')
                conf = det.get('confidence', 0.0)

                label = f"{class_name} {conf:.2f}"
                color = (0, 255, 0)

                cv2.rectangle(mask, (x1, y1), (x2, y2), color, 2)
                cv2.putText(mask, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # return mask 

        except Exception as e:
            print(f"[OBJ RESULT PROCESS ERROR] {e}")
            return None

        # "class_id": cls_id,
        # "class_name": class_names[cls_id],
        # "confidence": round(conf, 3),
        # "bbox": [x1, y1, x2, y2]

        # ['car', 'child_protection', 'construction', 'person', 'speed_limit_30', 
        # 'speed_limit_50', 'stop_sign', 'veh_go', 'veh_goLeft', 'veh_stop', 'veh_warning']

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("COVA II")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_gui)
        self.timer.start(33)  # ~30fps

        # 신호등과 정지선
        self.state = True # Moving
        self.prev_state = True

        self.udp_sender = UdpSender()
        self.tcp_lane_receiver = TcpLaneReceiver()
        self.tcp_obj_receiver = TcpObjReceiver()
        self.lane_result_processor = LaneResultProcessor()


        threading.Thread(target=self.udp_sender.send_frame, daemon=True).start()
        threading.Thread(target=self.tcp_lane_receiver.receive_data, daemon=True).start()
        threading.Thread(target=self.tcp_obj_receiver.receive_data, daemon=True).start()
        threading.Thread(target=self.lane_result_processor.process_result, daemon=True).start()

    def draw_result_on_frame(self, frame, result_json, obj_result):
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
                    color = (255, 255, 255) if lane["class_name"] == "white_solid" else (128, 128, 128)
                    for i in range(len(lane_pts) - 1):
                        pt1 = tuple(lane_pts[i])
                        pt2 = tuple(lane_pts[i + 1])
                        cv2.line(annotated, pt1, pt2, color, 2)

            # 정지선, 횡단보도 추가
            if "stop_line" in result_json:
                pts = result_json["stop_line"]
                for i in range(len(pts) - 1):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[i + 1])
                    cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)  # Red

            if "crosswalk" in result_json:
                pts = result_json["crosswalk"]
                for i in range(len(pts) - 1):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[i + 1])
                if pts and len(pts) > 2: # 점이 세 개 이상 있어야 다각형을 채울 수 있음
                    cv2.fillPoly(annotated, [np.array(pts, dtype=np.int32)], (0, 255, 0))  # Green


            if obj_result is not None and isinstance(obj_result, list):
                for det in obj_result:
                    x1, y1, x2, y2 = det["bbox"]
                    cls_name = det.get("class_name", str(det["class_id"]))
                    conf = det["confidence"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
            # # 정지선과의 거리 2.0m 미만이고 빨간불일떄 
            # self.state = True        
            # if result_json.get("real_distance") < 2.0 and obj_result.get("class_name") == "vehicle_stop":
            #     # print("STOP")
            #     self.state = False

            # if self.state==True and self.prev_state==False:
            #     # print("GO")
            #     self.prev_state = self.state

            # angle = result_json.get("steering_angle", 0.0)
            # cv2.putText(annotated, f"Steering Angle: {angle:.2f}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            print(f"[DRAW ERROR] {e}")

        return annotated
    
    def update_video_gui(self):
        frame = udp_video_queue.get() # uuid, frame

        lane_result = lane_result_queue.get() # uuid, pred_mask, msg

        obj_result = obj_tcp_queue.get() # uuid, overlay, cls_name


        if frame is not None:
            frame = self.draw_result_on_frame(frame, lane_result, obj_result)

            angle = lane_result.get("steering_angle", 0.0)
            self.label_msg_angle.setText(f"Angle: {angle:.2f}")

            real_distance = lane_result.get("real_distance", 0.0)
            self.state = True        
            if real_distance < 2.0 and obj_result.get("class_name") == "vehicle_stop":
                self.label_msg_alert.setText("STOP")
                self.state = False

            if self.state==True and self.prev_state==False:
                self.label_msg_alert.setText("GO")
                self.prev_state = self.state

            # 차선 변경 가능 유무 메세지
            self.label_msg_lane.setText("차선 변경이 가능합니다")



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
