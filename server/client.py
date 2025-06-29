import socket, json, cv2, threading, time, sys, queue, struct, base64, multiprocessing
import numpy as np

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

from threading import Lock

from distance import *

import pandas as pd

from multiprocessing import Process, Queue, Manager

from typing import Any

from db_Manager.db_connector import DBConnector
from db_Manager.db_inserter import *
from datetime import datetime



# 서버 IP 및 포트 정보
#LANE_SERVER_IP = "192.168.0.252"
LANE_SERVER_IP = "172.30.1.43"
TCP_LANE_PORT = 12345
UDP_LANE_PORT = 54321

#OBJ_SERVER_IP = "192.168.0.55"
OBJ_SERVER_IP = "172.30.1.86"
TCP_OBJ_PORT = 12346
UDP_OBJ_PORT = 54322

orig_frame = {}
lane_mask = {}
obj_mask = {}

frame_time = {}  # uuid: timestamp

HEADER_LENGTH = 4
UUID_LENGTH = 4

original_latency_check = {}
lane_latency_check = {}
obj_latency_check = {}

from_class = uic.loadUiType("/home/KTH/dl_project/gui/client_video.ui")[0]

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
        self.cap = cv2.VideoCapture("/home/KTH/Downloads/WIN_20250626_07_20_01_Pro.mp4")
        
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

                h, w = pred_mask.shape
                left_zone = pred_mask[int(h * 0.5):, int(w * 0.2):int(w * 0.4)]
                right_zone = pred_mask[int(h * 0.5):, int(w * 0.6):int(w * 0.8)]

                can_change_left = np.count_nonzero(left_zone == 2) > 40
                can_change_right = np.count_nonzero(right_zone == 2) > 40
                stop_line = np.count_nonzero(pred_mask == 4) > 50
                crosswalk = np.count_nonzero(pred_mask == 5) > 50

                msg = [0, 0, 0, 0]
                if can_change_left: msg[0] = 1
                if can_change_right: msg[1] = 1
                if stop_line: msg[2] = 1
                if crosswalk: msg[3] = 1

                lane_result_queue.put((uuid, pred_mask, msg))
                # print("lane: ", lane_result_queue.qsize())
                
                lane_latency_check[uuid] = {"lane": time.time()}

    except Exception as e:
        # print(f"[LANE RESULT PROCESS ERROR] {e}")
        # import traceback
        # traceback.print_exc()
        # continue
        return None

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
                    bbox_raw = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2)
                    bbox = {
                        bbox:bbox_raw
                    }
                    confidence = float(obj_data['confidence'])
                    class_name = obj_data.get('class_name', 'unknown')
                    label = f"{class_name}"
                    color = (0, 255, 0)

                    class_id = obj_data.get('class_id', -1)
                    obj_distance = estimate_obj_distance(class_id, (x1,y1,x2,y2))

                    cv2.rectangle(mask, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(mask, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                obj_result_queue.put((uuid, mask, class_id, obj_distance, confidence, bbox))
                # print("obj result : ", obj_result_queue.qsize())
                
                obj_latency_check[uuid] = {"obj": time.time()}

    except Exception as e:
        print(f"[OBJ RESULT PROCESS ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("COVA II")
        #self.setStyleSheet(open("/home/lee/dev_ws/projects/DL_project/final/server/dark_style.css").read())

        self.prev_lane_mask = None
        self.prev_obj_mask = None

        self.video_thread = VideoUpdateThread(self.label_video_lane.width(), self.label_video_lane.height())
        # 예: MainWindow 클래스에서 VideoUpdateThread 객체를 생성 후 연결
        

        # Signal 연결
        self.video_thread.lane_message.connect(self.label_msg_lane.setText)
        self.video_thread.alert_message.connect(self.label_msg_alert.setText)
        self.video_thread.obj_message.connect(self.label_msg_obj.setText)
        self.video_thread.state_message.connect(self.label_msg_state.setText)

        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.start()


        self.udp_sender = UdpSender()

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
    frame_ready = pyqtSignal(QPixmap)

    # 추가된 Signals (UI 업데이트용)
    lane_message = pyqtSignal(str)
    alert_message = pyqtSignal(str)
    obj_message = pyqtSignal(str)
    state_message = pyqtSignal(str)

    def __init__(self, label_width, label_height, parent=None):
        super().__init__(parent)
        self.label_width = label_width
        self.label_height = label_height
        self.running = True
        self.lane_msg = None
        self.obj_class = None

        self.MAX_WAIT_TIME = 0.3

        self.lane_color = {
            0: [0, 0, 0],
            1: [0, 0, 255],
            2: [0, 255, 255],
            3: [0, 255, 0],
            4: [255, 0, 0],
            5: [255, 0, 255],
        }

        self.lane_dist = None
        self.obj_dist = None
        self.prev_lane_mask = None
        self.prev_bbox = None

        # DB 삽입을 위한 데이터 수집
        self.session_id = None
        self.session_start_time = datetime.now()
        self.detected_objects = []
        self.action_logs = []
        self.total_distance = 0.0

        # DB 연결 및 inserter 초기화
        self.db_connector = DBConnector()
        #self.object_type_inserter = ObjectTypeInserter(self.db_connector)
        #self.action_type_inserter = ActionTypeInserter(self.db_connector)
        self.detected_object_inserter = DetectedObjectInserter(self.db_connector)
        self.action_log_inserter = ActionLogInserter(self.db_connector)
        self.drive_session_inserter = DriveSessionInserter(self.db_connector)

        try:
            self.session_id = self.drive_session_inserter.insert_drive_session(
                start_time=self.session_start_time,
                end_time=None,
                total_distance=0.0,
                result_summary="{}"
            )
            print(f"Drive session started with ID: {self.session_id}")
        except Exception as e:
            print(f"Failed to start drive session: {e}")

    def colorize_mask_lane(self, mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for k, color in self.lane_color.items():
            color_mask[mask == k] = color
        return color_mask

    def update_msg(self, msg, cls_id, obj_time):
        if msg is None or cls_id is None:
            return

        # ✅ 1. 차선변경 여부 (수정하지 않음)
        if msg[0] == 1 and cls_id != 0:
            self.lane_message.emit("좌측 차선 변경 가능")
            self.action_log_inserter.insert_action_log(
                cls_id + 1,
                3,
                datetime.now(),
                datetime.now() - obj_time,
                'Success'
            )
        elif msg[1] == 1 and cls_id != 0:
            self.lane_message.emit("우측 차선 변경 가능")
            self.action_log_inserter.insert_action_log(
                cls_id + 1,
                4,
                datetime.now(),
                datetime.now() - obj_time,
                'Success'
            )
        elif cls_id in (10, 9, 8, 7):
            self.lane_message.emit("차선 변경 불가능")
            self.action_log_inserter.insert_action_log(
                cls_id + 1,
                4,
                datetime.now(),
                datetime.now() - obj_time,
                'Success'
            )
        else:
            self.lane_message.emit("차선 변경 불가능")

        # ✅ 2. 정지선 (거리 3.5m 이내만 처리)
        if msg[2] == 1:
            self.lane_dist = estimate_lane_distance(self.prev_lane_mask, 0.05)
            if self.lane_dist is not None and self.lane_dist < 3.5 and cls_id == 9:
                self.state_message.emit("정지")
            elif self.lane_dist is not None and self.lane_dist < 3.5:
                self.state_message.emit("주행 중")

        # ✅ 3. 횡단보도 + 사람 (거리 3.5m 이내만 처리)
        if msg[3] == 1:
            if cls_id == 3:
                self.obj_dist = estimate_obj_distance(cls_id, self.prev_bbox)
                if self.obj_dist is not None and self.obj_dist < 3.5:
                    self.alert_message.emit("횡단보도에 사람이 있습니다. 주의하세요.")
                    if self.obj_dist < 3.0:
                        self.state_message.emit("정지")
                    else:
                        self.state_message.emit("주행 중")
            elif cls_id != 3:
                self.state_message.emit("주행 중")

        # ✅ 4. 객체별 (거리 3.5m 이내만 메시지 업데이트)
        if cls_id == 0:
            self.obj_dist = estimate_obj_distance(cls_id, self.prev_bbox)
            if self.obj_dist is not None and self.obj_dist < 3.5:
                self.obj_message.emit(f"자동차 인식됨 (거리: {self.obj_dist:.2f}m)")

        elif cls_id == 1:
            self.obj_dist = estimate_obj_distance(cls_id, self.prev_bbox)
            if self.obj_dist is not None and self.obj_dist < 3.5:
                self.alert_message.emit("어린이 보호구역 - 주의")
                self.obj_message.emit("어린이 보호구역")

        elif cls_id == 2:
            self.obj_dist = estimate_obj_distance(cls_id, self.prev_bbox)
            if self.obj_dist is not None and self.obj_dist < 3.5:
                self.alert_message.emit("공사장 근처 - 주의")
                self.obj_message.emit("공사장")

        elif cls_id == 4:
            self.obj_dist = estimate_obj_distance(cls_id, self.prev_bbox)
            if self.obj_dist is not None and self.obj_dist < 3.5:
                self.alert_message.emit("30km/h 이하로 주행")
                self.obj_message.emit("30km/h 속도제한 구간")

        elif cls_id == 5:
            self.obj_dist = estimate_obj_distance(cls_id, self.prev_bbox)
            if self.obj_dist is not None and self.obj_dist < 3.5:
                self.alert_message.emit("50km/h 이하로 주행")
                self.obj_message.emit("50km/h 속도제한 구간")

        elif cls_id == 6:
            self.obj_dist = estimate_obj_distance(cls_id, self.prev_bbox)
            if self.obj_dist is not None and self.obj_dist < 3.5:
                self.obj_message.emit("정지 표지판 인식됨")
                self.state_message.emit("정지")
                QTimer.singleShot(5000, lambda: self.state_message.emit("주행 중"))

    def run(self):
        while self.running:
            now = time.time()

            # 프레임 수신
            if not udp_video_queue.empty():
                udp_data = udp_video_queue.get_nowait()
                uuid = udp_data[0]
                frame = udp_data[1]
                orig_frame[uuid] = frame
                frame_time[uuid] = now

            while not lane_result_queue.empty():
                lane_data = lane_result_queue.get_nowait()
                lane_uuid = lane_data[0]
                lane_result = lane_data[1]
                self.lane_msg = lane_data[2]
                lane_mask[lane_uuid] = lane_result

            while not obj_result_queue.empty():
                obj_data = obj_result_queue.get_nowait()
                obj_uuid = obj_data[0]
                obj_result = obj_data[1]
                self.obj_class = obj_data[2]
                position = obj_data[3]
                confidence = obj_data[4]
                bbox = obj_data[5]
                obj_mask[obj_uuid] = obj_result
                detected_time = datetime.now()
                self.detected_object_inserter.insert_detected_object(
                    self.session_id,
                    self.obj_class+1,
                    detected_time,
                    confidence,
                    bbox,
                    position
                )

            try:
                self.update_msg(self.lane_msg, self.obj_class, detected_time)
            except Exception as e:
                print("update_msg 예외:", e)

            ready_uuids = sorted(orig_frame.keys())
            for uuid in ready_uuids:
                frame_age = now - frame_time.get(uuid, now)
                lane = lane_mask.get(uuid)
                obj = obj_mask.get(uuid)

                if lane is None or obj is None:
                    if frame_age < self.MAX_WAIT_TIME:
                        continue
                    else:
                        print(f"[WARN] {uuid}: 마스크 지연 - {frame_age:.2f}s → 부분 처리 진행")

                frame = orig_frame.pop(uuid)
                frame_time.pop(uuid, None)
                lane = lane_mask.pop(uuid, None)
                obj = obj_mask.pop(uuid, None)

                if lane is not None and len(lane.shape) == 2:
                    try:
                        lane = self.colorize_mask_lane(lane)
                        frame = cv2.addWeighted(frame, 0.7, lane, 0.3, 0.0)
                    except Exception:
                        pass

                if obj is not None:
                    try:
                        frame = cv2.addWeighted(frame, 0.7, obj, 0.3, 0.0)
                    except Exception:
                        pass

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                scaled = pixmap.scaled(self.label_width, self.label_height, Qt.AspectRatioMode.KeepAspectRatio)
                self.frame_ready.emit(scaled)

            expire_threshold = 2.0
            for mask_dict in (lane_mask, obj_mask):
                expired = [uuid for uuid in mask_dict if now - frame_time.get(uuid, now) > expire_threshold]
                for uuid in expired:
                    mask_dict.pop(uuid, None)

            self.lane_dist = None
            self.obj_dist = None

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


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
