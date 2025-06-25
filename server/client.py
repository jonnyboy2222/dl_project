import socket, json, cv2, threading, time, sys, queue, struct, base64, multiprocessing
import numpy as np


from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

from threading import Lock

from distance import estimate_distance

import pandas as pd

from multiprocessing import Process, Queue, Manager

from typing import Any



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

frame_time = {}  # uuid: timestamp

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

        self.fps_limit = 15
        self.frame_interval = 1 / self.fps_limit
        self.next_time = time.perf_counter()

    def send_frame(self):
        global original_latency_check

        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture("/home/lee/dev_ws/projects/DL_project/final/server/lane.avi")
        
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

def receive_tcp_lane(server_ip, server_port, output_queue):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        sock.settimeout(5.0)

        def recv_exact(sock, num_bytes):
            buffer = b''
            while len(buffer) < num_bytes:
                chunk = sock.recv(num_bytes - len(buffer))
                if not chunk:
                    raise ConnectionError("Socket closed before expected data received")
                buffer += chunk
            return buffer

        while True:
            try:
                header = recv_exact(sock, 4)
                json_len = struct.unpack('>I', header)[0]

                uuid_raw = recv_exact(sock, 4)
                uuid = struct.unpack('>I', uuid_raw)[0]

                buffer = recv_exact(sock, json_len)
                result = json.loads(buffer.decode('utf-8'))

                if "pred_mask" not in result:
                    continue

                # base64 디코딩
                encoded = result["pred_mask"]
                data = base64.b64decode(encoded)
                nparr = np.frombuffer(data, np.uint8)
                pred_mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                output_queue.put((uuid, pred_mask))
                # print(output_queue.qsize())
                
            except Exception as e:
                print(f"[receive_tcp_lane ERROR] {e}")
                continue
    finally:
        sock.close()

# class TcpLaneReceiver(Process):
#     def __init__(self, server_ip, server_port, output_queue):
#         super().__init__()
#         self.server_ip = server_ip
#         self.server_port = server_port
#         self.output_queue = output_queue
#         self.sock = None

#     @staticmethod
#     def decode_mask_png_base64(encoded: str) -> np.ndarray:
#         data = base64.b64decode(encoded)
#         nparr = np.frombuffer(data, np.uint8)
#         mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
#         return mask

#     @staticmethod
#     def recv_exact(sock, num_bytes):
#         buffer = b''
#         while len(buffer) < num_bytes:
#             chunk = sock.recv(num_bytes - len(buffer))
#             if not chunk:
#                 raise ConnectionError("Socket closed before expected data received")
#             buffer += chunk
#         return buffer

#     def run(self):
#         try:
#             self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.sock.connect((self.server_ip, self.server_port))
#             self.sock.settimeout(5.0)

#             while True:
#                 try:
#                     # Header (4 bytes)
#                     header = self.recv_exact(self.sock, 4)
#                     json_len = struct.unpack('>I', header)[0]

#                     # UUID (4 bytes)
#                     uuid_raw = self.recv_exact(self.sock, 4)
#                     uuid = struct.unpack('>I', uuid_raw)[0]

#                     # Payload
#                     payload = self.recv_exact(self.sock, json_len)
#                     result = json.loads(payload.decode('utf-8'))

#                     if "pred_mask" not in result:
#                         print(f"[WARN] Missing 'pred_mask' for UUID {uuid}")
#                         continue

#                     pred_mask = self.decode_mask_png_base64(result["pred_mask"])
#                     self.output_queue.put((uuid, pred_mask))

#                 except Exception as e:
#                     print(f"[TcpLaneReceiver ERROR] {e}")
#                     continue

#         finally:
#             if self.sock:
#                 self.sock.close()


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

def receive_tcp_obj(server_ip, server_port, output_queue: Queue):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))
        sock.settimeout(5.0)

        def recv_exact(sock, num_bytes):
            buffer = b''
            while len(buffer) < num_bytes:
                chunk = sock.recv(num_bytes - len(buffer))
                if not chunk:
                    raise ConnectionError("Socket closed before expected data received")
                buffer += chunk
            return buffer

        while True:
            try:
                header = recv_exact(sock, 4)
                # print(len(header))
                json_len = struct.unpack('>I', header)[0]

                uuid_raw = recv_exact(sock, 4)
                uuid = struct.unpack('>I', uuid_raw)[0]

                buffer = recv_exact(sock, json_len)
                json_data = json.loads(buffer.decode('utf-8'))

                output_queue.put((uuid, json_data))
                # print(output_queue.qsize())

            except Exception as e:
                print(f"[receive_tcp_obj ERROR] {e}")
                continue

    finally:
        sock.close()

# class TcpObjReceiver(Process):
#     def __init__(self, server_ip: str, server_port: int, output_queue: Any):
#         super().__init__()
#         self.server_ip = server_ip
#         self.server_port = server_port
#         self.output_queue = output_queue
#         self.sock = None

#     @staticmethod
#     def recv_exact(sock, num_bytes: int) -> bytes:
#         buffer = b''
#         while len(buffer) < num_bytes:
#             chunk = sock.recv(num_bytes - len(buffer))
#             if not chunk:
#                 raise ConnectionError("Socket closed before expected data received")
#             buffer += chunk
#         return buffer

#     def run(self):
#         try:
#             self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.sock.connect((self.server_ip, self.server_port))
#             self.sock.settimeout(5.0)

#             while True:
#                 try:
#                     # Receive 4-byte header
#                     header = self.recv_exact(self.sock, 4)
#                     json_len = struct.unpack('>I', header)[0]

#                     # Receive 4-byte UUID
#                     uuid_raw = self.recv_exact(self.sock, 4)
#                     uuid = struct.unpack('>I', uuid_raw)[0]

#                     # Receive JSON payload
#                     buffer = self.recv_exact(self.sock, json_len)
#                     json_data = json.loads(buffer.decode('utf-8'))

#                     self.output_queue.put((uuid, json_data))

#                 except Exception as e:
#                     print(f"[TcpObjReceiver ERROR] {e}")
#                     continue

#         finally:
#             if self.sock:
#                 self.sock.close()

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

        self.prev_lane_mask = None
        self.prev_obj_mask = None

        self.video_thread = VideoUpdateThread(self.label_video_lane.width(), self.label_video_lane.height())
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()

        # 신호등과 정지선
        # self.distance = np.inf
        # self.red_detected = False
        # self.state = True # Moving
        # self.prev_state = True

        self.udp_sender = UdpSender()
        # self.tcp_lane_receiver = TcpLaneReceiver()
        # self.tcp_obj_receiver = TcpObjReceiver()
        # self.lane_result_processor = LaneResultProcessor()
        # self.obj_result_processor = ObjectResultProcessor()
        self.lane_recv = Process(target=receive_tcp_lane,
                                args=(LANE_SERVER_IP, TCP_LANE_PORT, lane_tcp_queue))
        self.obj_recv = Process(target=receive_tcp_obj,
                                args=(OBJ_SERVER_IP, TCP_OBJ_PORT, obj_tcp_queue))
        
        self.lane_recv.start()
        self.obj_recv.start()

        self.lane_process = Process(target=lane_result_worker,
                                    args=(lane_tcp_queue, lane_result_queue, lane_latency_check))
        self.obj_process = Process(target=obj_result_worker,
                                args=(obj_tcp_queue, obj_result_queue, obj_latency_check))

        self.lane_process.start()
        self.obj_process.start()

        threading.Thread(target=self.udp_sender.send_frame, daemon=True).start()
        # threading.Thread(target=self.tcp_lane_receiver.receive_data, daemon=True).start()
        # threading.Thread(target=self.tcp_obj_receiver.receive_data, daemon=True).start()
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
    
    def update_frame(self, pixmap: QPixmap):
        self.label_video_lane.setPixmap(pixmap)

    def closeEvent(self, event):
        self.video_thread.stop()
        
        self.udp_sender.close()

        self.lane_process.terminate()
        self.obj_process.terminate()
        self.lane_process.join()
        self.obj_process.join()

        self.lane_recv.terminate()
        self.obj_recv.terminate()
        self.lane_recv.join()
        self.obj_recv.join()

        event.accept()

class VideoUpdateThread(QThread):
    frame_ready = pyqtSignal(QPixmap)  # signal to GUI

    def __init__(self, label_width, label_height, parent=None):
        super().__init__(parent)
        # print("video update thread")
        self.label_width = label_width
        self.label_height = label_height
        self.running = True

    def run(self):
        while self.running:
            # 1. 프레임 수신
            if not udp_video_queue.empty():
                udp_data = udp_video_queue.get_nowait()
                uuid = udp_data[0]
                frame = udp_data[1]
                print("original", uuid)
                orig_frame[uuid] = frame
                frame_time[uuid] = time.time()

            # 2. 마스크 수신
            while not lane_result_queue.empty():
                lane_data = lane_result_queue.get_nowait()
                lane_uuid = lane_data[0]
                lane_result = lane_data[1]
                lane_mask[lane_uuid] = lane_result
                print("lane :", lane_uuid)

            while not obj_result_queue.empty():
                obj_data = obj_result_queue.get_nowait()
                obj_uuid = obj_data[0]
                obj_result = obj_data[1]
                obj_mask[obj_uuid] = obj_result
                print("obj :", obj_uuid)

            # 3. 처리 가능한 프레임 추출 (도착 순서 기준)
            ready_uuids = sorted(orig_frame.keys())  # 순차적으로 처리
            for uuid in ready_uuids:
                frame = orig_frame.pop(uuid)
                frame_time.pop(uuid, None)
                lane = lane_mask.pop(uuid, None)
                # print(lane.shape)
                obj = obj_mask.pop(uuid, None)
                # print(obj.shape)

                if lane is not None and len(lane.shape) == 2:
                    print("check1")
                    try:
                        lane = self.colorize_mask_lane(lane)
                        frame = cv2.addWeighted(frame, 0.8, lane, 0.2, 0.0)
                    except Exception:
                        print("lane none")
                        pass

                if obj is not None:
                    print("check2")
                    try:
                        frame = cv2.addWeighted(frame, 0.8, obj, 0.2, 0.0)
                    except Exception:
                        print("obj none")
                        pass

                # Qt 변환 및 emit
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                scaled = pixmap.scaled(self.label_width, self.label_height, Qt.AspectRatioMode.KeepAspectRatio)
                self.frame_ready.emit(scaled)

            # 4. 오래된 마스크 제거 (메모리 누수 방지)
            now = time.time()
            expire_threshold = 2.0
            for mask_dict in (lane_mask, obj_mask):
                expired = [uuid for uuid in mask_dict if now - frame_time.get(uuid, now) > expire_threshold]
                for uuid in expired:
                    mask_dict.pop(uuid, None)

        
    # def run(self):
    #     # global udp_video_queue, orig_frame, lane_mask, obj_mask
    #     prev_lane_mask = None
    #     prev_obj_mask = None

    #     while self.running:
    #         if udp_video_queue.empty():
    #             # time.sleep(0.005)
    #             continue
            
    #         udp_data = udp_video_queue.get_nowait()

    #         uuid = udp_data[0]
    #         print("original", uuid)
    #         frame = udp_data[1]

    #         orig_frame[uuid] = frame
    #         lane = None
    #         obj = None

    #         while not lane_result_queue.empty():
    #             lane_result = lane_result_queue.get_nowait()
    #             lane_mask[lane_result[0]] = lane_result[1]
    #             print("lane : ", lane_result[0])

    #         while not obj_result_queue.empty():
    #             obj_result = obj_result_queue.get_nowait()
    #             obj_mask[obj_result[0]] = obj_result[1]
    #             print("obj : ", obj_result[0])
    #             # print(len(obj_result[1]))
    #         # print(obj_mask.keys())

    #         # frame = frame.copy()
    #         frame = orig_frame.pop(uuid)

    #         # Lane mask 처리
    #         if uuid in lane_mask:
    #             lane = lane_mask.pop(uuid)
    #             # print(len(lane))
    #             if len(lane.shape) == 2:
    #                 lane = self.colorize_mask_lane(lane)
    #                 prev_lane_mask = lane
    #         elif prev_lane_mask is not None:
    #             lane = prev_lane_mask
            
            

    #         try:
    #             if lane is not None:
    #                 frame = cv2.addWeighted(frame, 0.8, lane, 0.2, 0.0)
    #         except Exception:
    #             # import traceback
    #             # traceback.print_exc()
    #             pass

    #         # Object mask 처리
    #         if uuid in obj_mask:
    #             obj = obj_mask.pop(uuid)
    #             prev_obj_mask = obj
    #         elif prev_obj_mask is not None:
    #             obj = prev_obj_mask

    #         try:
    #             if obj is not None:
    #                 frame = cv2.addWeighted(frame, 0.8, obj, 0.2, 0.0)
    #         except Exception:
    #             # import traceback
    #             # traceback.print_exc()
    #             pass

    #         # Qt 이미지 변환 및 전송
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         h, w, ch = rgb_frame.shape
    #         bytes_per_line = ch * w
    #         img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    #         pixmap = QPixmap.fromImage(img)
    #         scaled = pixmap.scaled(self.label_width, self.label_height, Qt.AspectRatioMode.KeepAspectRatio)
    #         self.frame_ready.emit(scaled)

            # uuid 캐시 제한
            # MAX_CACHE = 100
            # if len(orig_frame) > MAX_CACHE:
            #     oldest_uuid = next(iter(orig_frame))
            #     del orig_frame[oldest_uuid]
            #     lane_mask.pop(oldest_uuid, None)
            #     obj_mask.pop(oldest_uuid, None)

            # time.sleep(0.01)

    def colorize_mask_lane(self, mask: np.ndarray) -> np.ndarray:
        # 예시: binary mask → RGB color map
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask > 0] = [255, 0, 255]
        return color_mask

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

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
