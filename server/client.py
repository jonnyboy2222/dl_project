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

from distance import estimate_distance

import base64
import pandas as pd

from multiprocessing import Process, Queue, Manager

import multiprocessing



# 서버 IP 및 포트 정보
LANE_SERVER_IP = "192.168.0.252"
TCP_LANE_PORT = 12345
UDP_LANE_PORT = 54321

OBJ_SERVER_IP = "192.168.0.22"
TCP_OBJ_PORT = 12346
UDP_OBJ_PORT = 54322



# lane_result_queue = queue.Queue()
# obj_result_queue = queue.Queue()



orig_frame = {}
lane_mask = {}
obj_mask = {}

HEADER_LENGTH = 4
UUID_LENGTH = 4

original_latency_check = {}
lane_latency_check = {}
obj_latency_check = {}


from_class = uic.loadUiType("/home/lee/dev_ws/projects/DL_project/final/gui/client_video.ui")[0]

class UdpSender():
    def __init__(self):
        self.udp_lane = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.uuid_counter = 0

        self.fps_limit = 30
        self.frame_interval = 1 / self.fps_limit
        self.next_time = time.perf_counter()

    def send_frame(self):
        global original_latency_check

        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture("/home/lee/dev_ws/projects/DL_project/final/server/lane.avi")
        
        try:
            if not self.cap.isOpened():
                print("[UDP] Webcam open failed")
                return

            while True:
                # 프레임 조절
                now = time.perf_counter()

                if now < self.next_time:
                    time.sleep(self.next_time - now)
                self.next_time += self.frame_interval
                
                # uuid
                self.uuid_counter += 1
                uuid_msg = self.uuid_counter.to_bytes(4, byteorder='big')

                # 프레임 캡처
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    continue

                frame = cv2.resize(frame, (512, 256))

                if not ret:
                    continue
                
                ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                
                if not ret:
                    continue

                try:
                    self.udp_lane.sendto(uuid_msg + b'||' + buffer.tobytes(), (LANE_SERVER_IP, UDP_LANE_PORT))
                    self.udp_obj.sendto(uuid_msg + b'||' + buffer.tobytes(), (OBJ_SERVER_IP, UDP_OBJ_PORT))

                    # time.sleep(0.05)
                    # print("origin:", type(frame)) # debug
                    udp_video_queue.put((self.uuid_counter, frame.copy()))

                    # latency_check
                    original_latency_check[self.uuid_counter] = {"start":time.time()}
                    # print(original_latency_check)

                except Exception as e:
                    # print(f"[UDP SEND ERROR] {e}")
                    # import traceback
                    # traceback.print_exc()
                    return None
        finally:
            self.cap.release()
            cv2.destroyAllWindows()



    def close(self):
        self.udp_lane.close()
        self.udp_obj.close()
        self.cap.release()

class TcpLaneReceiver():
    def __init__(self):
        self.tcp_lane = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_lane.connect((LANE_SERVER_IP, TCP_LANE_PORT))
        self.tcp_lane.settimeout(5.0)

    @staticmethod
    def decode_mask_png_base64(encoded: str) -> np.ndarray:
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return mask  # dtype=np.uint8, shape=(H, W)
    
    @staticmethod
    def recv_exact(sock, num_bytes):
        buffer = b''
        while len(buffer) < num_bytes:
            chunk = sock.recv(num_bytes - len(buffer))
            if not chunk:
                raise ConnectionError("Socket closed before expected data received")
            buffer += chunk
        return buffer    

    def receive_data(self):
        while True:
            try:
                # 정확히 4바이트 헤더 수신
                # print("[TCP LANE] Waiting for header...")
                header = self.recv_exact(self.tcp_lane, HEADER_LENGTH)
                json_len = struct.unpack('>I', header)[0]

                # 정확히 4바이트 UUID 수신
                uuid_raw = self.recv_exact(self.tcp_lane, UUID_LENGTH)
                uuid = struct.unpack('>I', uuid_raw)[0]

                # print(f"[TCP LANE] Receiving JSON payload (uuid={uuid}, len={json_len})")
                buffer = self.recv_exact(self.tcp_lane, json_len)

                try:
                    result = json.loads(buffer.decode('utf-8'))
                    # print("result decoding success")
                except Exception as e:
                    print(f"[TCP LANE][JSON ERROR] uuid={uuid} decode failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                if "pred_mask" not in result:
                    print(f"[TCP LANE][WARN] pred_mask not in result for UUID {uuid}")
                    continue

                pred_mask = self.decode_mask_png_base64(result["pred_mask"])
                # print("pred mask ready")
                # print("uuid : ", uuid, "pred_mask : ", len(pred_mask))
                lane_tcp_queue.put((uuid, pred_mask))
                # print("lane tcp : ", lane_tcp_queue.qsize())
                # print("queue insert")

            except Exception as e:
                print(f"[TCP LANE RECEIVE ERROR] {e}")
                continue
            
    def close(self):
        if self.tcp_lane is not None:
            self.tcp_lane.close()
            self.tcp_lane = None

def lane_result_worker(lane_tcp_queue, lane_result_queue, lane_latency_check):
    try:
        while True:
            if not lane_tcp_queue.empty():
                # print("queue check")
                lane_data = lane_tcp_queue.get_nowait()
                uuid = lane_data[0]
                # print("uuid : ", uuid)
                # print("lane uuid : ", uuid) # debug
                pred_mask = lane_data[1]

                if pred_mask is None:
                    continue

                # print("queue check")
                # try:
                #     lane_data = lane_tcp_queue.get()
                # except queue.Empty:
                #     print("queue empty called")
                #     continue
                # print("queue extract")



                # uuid = lane_data[0]
                # pred_mask = lane_data[1]

                # print("lane uuid : ", uuid) # debug

                # if pred_mask is None:
                #     return

                h, w = pred_mask.shape
                left_zone = pred_mask[int(h * 0.5):, int(w * 0.2):int(w * 0.4)]
                right_zone = pred_mask[int(h * 0.5):, int(w * 0.6):int(w * 0.8)]

                can_change_left = np.count_nonzero(left_zone == 2) > 40
                can_change_right = np.count_nonzero(right_zone == 2) > 40
                stop_line = np.count_nonzero(pred_mask == 4) > 50
                crosswalk = np.count_nonzero(pred_mask == 5) > 50

                msg = [0, 0, 0, 0, 0]
                if can_change_left: msg[0] = 1
                if can_change_right: msg[1] = 1
                if not (can_change_left or can_change_right): msg[2] = 1
                if stop_line: msg[3] = 1
                if crosswalk: msg[4] = 1

                lane_result_queue.put((uuid, pred_mask, msg))
                # print("lane: ", lane_result_queue.qsize())
                
                lane_latency_check[uuid] = {"lane": time.time()}

    except Exception as e:
        # print(f"[LANE RESULT PROCESS ERROR] {e}")
        # import traceback
        # traceback.print_exc()
        # continue
        return None

# class LaneResultProcessor():
#     def __init__(self):
#         pass

#     def process_result(self):
#         try:
#             while True:
#                 if not lane_tcp_queue.empty():
#                     lane_data = lane_tcp_queue.get_nowait()
#                     uuid = lane_data[0]
#                     # print("lane uuid : ", uuid) # debug
#                     pred_mask = lane_data[1]
                    
#                     if pred_mask is None:
#                         continue

#                     h, w = pred_mask.shape

#                     left_zone = pred_mask[int(h * 0.5):, int(w * 0.2):int(w * 0.4)]
#                     right_zone = pred_mask[int(h * 0.5):, int(w * 0.6):int(w * 0.8)]

#                     can_change_left = np.count_nonzero(left_zone == 2) > 40
#                     can_change_right = np.count_nonzero(right_zone == 2) > 40

#                     stop_line = np.count_nonzero(pred_mask == 4) > 50
#                     crosswalk = np.count_nonzero(pred_mask == 5) > 50

#                     # print(pred_mask) # debug

#                     msg = [0, 0, 0, 0, 0]
#                     if can_change_left:
#                         msg[0] = 1
#                     if can_change_right:
#                         msg[1] = 1
#                     if not (can_change_left or can_change_right):
#                         msg[2] = 1
#                     if stop_line:
#                         msg[3] = 1
#                     if crosswalk:
#                         msg[4] = 1

#                     lane_result_queue.put((uuid, pred_mask, msg))

#                     # latency_check
#                     lane_latency_check[uuid] = {"lane":time.time()}
#                     # print("lane2 latency") # debug
#                     # return uuid, pred_mask

#         except Exception as e:
#             print(f"[LANE RESULT PROCESS ERROR] {e}")
#             return None


class TcpObjReceiver():
    def __init__(self):
        self.tcp_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_obj.connect((OBJ_SERVER_IP, TCP_OBJ_PORT))
        self.tcp_obj.settimeout(5.0)

    @staticmethod
    def recv_exact(sock, num_bytes):
        buffer = b''
        while len(buffer) < num_bytes:
            chunk = sock.recv(num_bytes - len(buffer))
            if not chunk:
                raise ConnectionError("Socket closed before expected data received")
            buffer += chunk
        return buffer


    def receive_data(self):
        while True:
            try:
                # print("[TCP OBJ] waiting for header")
                header = self.recv_exact(self.tcp_obj, HEADER_LENGTH)
                # print("[TCP OBJ] received header")

                json_len = struct.unpack('>I', header)[0]

                uuid_raw = self.recv_exact(self.tcp_obj, UUID_LENGTH)
                uuid = struct.unpack('>I', uuid_raw)[0]
                # print(f"[TCP OBJ] uuid={uuid}, expecting {json_len} bytes")

                buffer = self.recv_exact(self.tcp_obj, json_len)

                json_data = json.loads(buffer.decode('utf-8'))
                obj_tcp_queue.put((uuid, json_data))
                # print("obj queue insert : ", obj_tcp_queue.qsize())

            except Exception as e:
                print(f"[TCP OBJ RECEIVE ERROR] {e}")
                continue

            
    def close(self):
        if self.tcp_obj is not None:
            self.tcp_obj.close()
            self.tcp_obj = None

def obj_result_worker(obj_tcp_queue, obj_result_queue, obj_latency_check):
    try:
        while True:
            if not obj_tcp_queue.empty():
                # print("queue check")
                obj_tcp_data = obj_tcp_queue.get_nowait()
                # print("queue extract")
                uuid = obj_tcp_data[0]
                # print("obj uuid : ", uuid) # debug

                mask = np.zeros((256, 512, 3), dtype=np.uint8)

                class_id = -1
                for obj_data in obj_tcp_data[1]:
                    if not isinstance(obj_data["bbox"], list):
                        continue

                    x1, y1, x2, y2 = obj_data['bbox']
                    class_name = obj_data.get('class_name', 'unknown')
                    label = f"{class_name}"
                    color = (0, 255, 0)

                    class_id = obj_data.get('class_id', -1)

                    cv2.rectangle(mask, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(mask, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                obj_result_queue.put((uuid, mask, class_id))
                # print("obj result : ", obj_result_queue.qsize())
                
                obj_latency_check[uuid] = {"obj": time.time()}

    except Exception as e:
        print(f"[OBJ RESULT PROCESS ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


# class ObjectResultProcessor():
#     def __init__(self):
#         pass

#     def process_result(self):
#         try:
#             # 큐에서 최신 결과 추출
#             while True:
#                 if not obj_tcp_queue.empty():
#                     # print("check")
#                     obj_tcp_data = obj_tcp_queue.get_nowait()

#                     uuid = obj_tcp_data[0]
#                     # print("obj uuid : ", uuid) # debug

#                     mask = np.zeros((256, 512, 3), dtype=np.uint8)

#                     for obj_data in obj_tcp_data[1]:
#                         if not isinstance(obj_data["bbox"], list):
#                             print('obj is none')
#                             continue
                        
#                         x1, y1, x2, y2 = obj_data['bbox']
#                         class_id = obj_data.get('class_id', -1)
#                         class_name = obj_data.get('class_name', 'unknown')

#                         label = f"{class_name}"
#                         color = (0, 255, 0)

#                         cv2.rectangle(mask, (x1, y1), (x2, y2), color, 2)
#                         cv2.putText(mask, label, (x1, y1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#                     # print("obj: ",type(mask)) # debug
#                     obj_result_queue.put((uuid, mask, class_id))

#                     # latency_check
#                     obj_latency_check[uuid] = {"obj":time.time()}
#                     # print("obj result")
#                     # return mask 

#         except Exception as e:
#             print(f"[OBJ RESULT PROCESS ERROR] {e}")
#             import traceback
#             traceback.print_exc()
#             return None

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("COVA II")
        self.setStyleSheet(open("/home/lee/dev_ws/projects/DL_project/final/server/dark_style.css").read())

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_gui)
        self.timer.start(33)  # ~30fps

        self.prev_lane_mask = None
        self.prev_obj_mask = None

        # 신호등과 정지선
        self.distance = np.inf
        self.red_detected = False
        self.state = True # Moving
        self.prev_state = True

        self.udp_sender = UdpSender()
        self.tcp_lane_receiver = TcpLaneReceiver()
        self.tcp_obj_receiver = TcpObjReceiver()
        # self.lane_result_processor = LaneResultProcessor()
        # self.obj_result_processor = ObjectResultProcessor()

        self.lane_process = Process(target=lane_result_worker,
                                    args=(lane_tcp_queue, lane_result_queue, lane_latency_check))
        self.obj_process = Process(target=obj_result_worker,
                                args=(obj_tcp_queue, obj_result_queue, obj_latency_check))

        self.lane_process.start()
        self.obj_process.start()

        threading.Thread(target=self.udp_sender.send_frame, daemon=True).start()
        threading.Thread(target=self.tcp_lane_receiver.receive_data, daemon=True).start()
        threading.Thread(target=self.tcp_obj_receiver.receive_data, daemon=True).start()
        # threading.Thread(target=self.lane_result_processor.process_result, daemon=True).start()
        # threading.Thread(target=self.obj_result_processor.process_result, daemon=True).start()

        self.lane_color = {
            0: [0, 0, 0],         # 배경 - 검정
            1: [0, 0, 255],       # 흰색 실선 - 선명한 빨강
            2: [0, 255, 255],     # 흰색 점선 - 시안 (cyan)
            3: [0, 255, 0],       # 중앙선(노란 실선) - 선명한 초록
            4: [255, 0, 0],       # 정지선 - 파랑
            5: [255, 0, 255],     # 횡단보도 - 마젠타
        }


    def colorize_mask_lane(self, mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for k, color in self.lane_color.items():
            color_mask[mask == k] = color
        return color_mask
    
    from PyQt6.QtCore import QCoreApplication  # GUI event loop 처리

    def update_video_gui(self):
        global orig_frame, lane_mask, obj_mask

        if udp_video_queue.empty():
            return

        uuid, frame = udp_video_queue.get_nowait()
        orig_frame[uuid] = frame
        lane = None
        obj = None

        # 수신된 결과들 업데이트
        while not lane_result_queue.empty():
            lane_result = lane_result_queue.get_nowait()
            lane_mask[lane_result[0]] = lane_result[1]

        while not obj_result_queue.empty():
            obj_result = obj_result_queue.get_nowait()
            obj_mask[obj_result[0]] = obj_result[1]

        # 결과가 아직 없을 경우 대기 (최대 200ms)
        start_time = time.time()
        while uuid not in lane_mask or uuid not in obj_mask:
            if time.time() - start_time > 0.2:
                print(f"[SKIP] uuid={uuid} 결과 미도착, 프레임 렌더링 생략")
                return
            QCoreApplication.processEvents()  # GUI hang 방지
            time.sleep(0.01)  # 10ms 간격으로 재확인

        # 여기까지 왔으면 결과 있음 → 렌더링 진행
        frame = frame.copy()

        lane = lane_mask.get(uuid)
        if lane is not None:
            if len(lane.shape) == 2:
                lane = self.colorize_mask_lane(lane)
                self.prev_lane_mask = lane
            frame = cv2.addWeighted(frame, 0.8, lane, 0.2, 0.0)
        elif self.prev_lane_mask is not None:
            frame = cv2.addWeighted(frame, 0.8, self.prev_lane_mask, 0.2, 0.0)

        obj = obj_mask.get(uuid)
        if obj is not None:
            self.prev_obj_mask = obj
            frame = cv2.addWeighted(frame, 0.8, obj, 0.2, 0.0)
        elif self.prev_obj_mask is not None:
            frame = cv2.addWeighted(frame, 0.8, self.prev_obj_mask, 0.2, 0.0)

        # Qt GUI 렌더링
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.label_video_lane.setPixmap(pixmap.scaled(
            self.label_video_lane.width(), self.label_video_lane.height(), Qt.AspectRatioMode.KeepAspectRatio))
        
    def closeEvent(self, event):
        
        self.udp_sender.close()
        self.tcp_lane_receiver.close()
        self.tcp_obj_receiver.close()

        self.lane_process.terminate()
        self.obj_process.terminate()
        self.lane_process.join()
        self.obj_process.join()

        event.accept()

    # def update_video_gui(self):
    #     global orig_frame, lane_mask, obj_mask

    #     uuid = None

    #     if not udp_video_queue.empty():
    #         uuid, frame = udp_video_queue.get_nowait()
    #         orig_frame[uuid] = frame
    #         lane = None
    #         obj = None

    #         if not lane_result_queue.empty():
    #             while not lane_result_queue.empty():
    #                 lane_result = lane_result_queue.get_nowait()
    #                 lane_mask[lane_result[0]] = lane_result[1]
    #             # lane = lane_mask.get(uuid)

    #         if not obj_result_queue.empty():
    #             while not obj_result_queue.empty():
    #                 obj_result = obj_result_queue.get_nowait()
    #                 obj_mask[obj_result[0]] = obj_result[1]
    #             # obj = obj_mask.get(uuid)

    #     if uuid is not None and uuid in orig_frame:
    #         # print("orig uuid : ", uuid)
    #         frame = orig_frame[uuid].copy()
    #         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #         if uuid in list(lane_mask.keys()):
    #             # print("lane uuid : ", uuid)
    #             lane = lane_mask[uuid]
    #             # print(np.unique(lane_mask[uuid]))
    #             if len(lane.shape) == 2:  # grayscale mask
    #                 lane = self.colorize_mask_lane(lane)
    #                 self.prev_lane_mask = lane
    #             frame = cv2.addWeighted(frame, 0.8, lane, 0.2, 0.0)

    #         elif self.prev_lane_mask is not None:
    #             lane = self.prev_lane_mask

    #             frame = cv2.addWeighted(frame, 0.8, lane, 0.2, 0.0)

    #         if uuid in list(obj_mask.keys()):
    #             self.red_detected = False
    #             obj = obj_mask[uuid]
                
    #             if estimate_distance(obj, 0.05, 9) <= 3.0:
    #                 self.red_detected = True

    #             self.prev_obj_mask = obj

    #             frame = cv2.addWeighted(frame, 0.8, obj, 0.2, 0.0)

    #         elif self.prev_obj_mask is not None:
    #             obj = self.prev_obj_mask

    #             frame = cv2.addWeighted(frame, 0.8, obj, 0.2, 0.0)

    #         # OpenCV BGR → Qt RGB
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         h, w, ch = rgb_frame.shape
    #         bytes_per_line = ch * w

    #         # QImage 생성 및 QLabel에 설정
    #         img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    #         pixmap = QPixmap.fromImage(img)
    #         self.label_video_lane.setPixmap(pixmap.scaled(
    #             self.label_video_lane.width(), self.label_video_lane.height(), Qt.AspectRatioMode.KeepAspectRatio))
            
    # def alert_msg(self):


    # def state_msg(self):
    #     if self.red_detected:
    #         if estimate_distance(obj_mask[uuid], 0.05, 4) <= 2.0:
    #             self.label_msg_state.setText("Stop")
    #             self.state = False


    #     self.prev_state = self.state

    # def object_msg(self):

    # def lane_msg(self):

    # def closeEvent(self, event):
    #     self.udp_sender.close()
    #     self.tcp_lane_receiver.close()
    #     self.tcp_obj_receiver.close()
    #     event.accept() # 창 닫기 허용

# Main
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    manager = Manager()

    udp_video_queue = manager.Queue()

    lane_tcp_queue = manager.Queue()
    obj_tcp_queue = manager.Queue()

    lane_result_queue = manager.Queue()
    obj_result_queue = manager.Queue()

    df_orig = pd.DataFrame.from_dict(original_latency_check, orient='index')
    df_lane = pd.DataFrame.from_dict(lane_latency_check, orient='index')
    df_obj  = pd.DataFrame.from_dict(obj_latency_check, orient='index')

    df_merged = pd.concat([df_orig, df_lane, df_obj], axis=1)

    df_merged.to_csv("latency_summary.csv", index_label="uuid")

    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    sys.exit(app.exec())
