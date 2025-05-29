import socket
import json
import cv2
import threading
import time
import numpy as np
import sys
from queue import Queue

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import uic

# 서버 IP 주소

SERVER_IP = '192.168.2.232'
TCP_SERVER_PORT = 12345
UDP_SERVER_PORT = 54321  # UDP 포트

from_class = uic.loadUiType("../gui/client_video.ui")[0]

class TcpClientThread(QThread):
    msg_from_server = pyqtSignal(str)
    #con_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            try:
                client.connect((SERVER_IP, TCP_SERVER_PORT))
                print("[TCP CLIENT] Connected to server.")
                client.settimeout(1.0)  # recv 호출에 대한 타임아웃 설정 (1초)

                message_counter = 0
                while self.running:
                    message_counter += 1
                    # 서버에 JSON 메시지 전송
                    message_to_send = {"type": "tcp_request", 
                                       "id": message_counter, 
                                       "data": "Hello from TCP client"}
                    try:
                        client.sendall(json.dumps(message_to_send).encode('utf-8'))
                        #print(f"[TCP CLIENT] Sent: {message_to_send}")

                        # 서버로부터 응답 수신
                        data = client.recv(1024) # 타임아웃 적용
                        if not data:
                            #print("[TCP CLIENT] Server closed connection.")
                            break
                    
                        response_message = json.loads(data.decode('utf-8'))
                        if "message" in response_message:
                            self.msg_from_server.emit(response_message["message"])
                        else:
                            self.msg_from_server.emit(str(response_message))

                    except socket.timeout:
                        # print("[TCP CLIENT] recv timed out. Checking shutdown signal.")
                        continue # 루프를 계속하여 shutdown_event 확인
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                        print(f"[TCP CLIENT ERROR] Connection error: {e}")
                        break
                    except json.JSONDecodeError as e:
                        print(f"[TCP CLIENT ERROR] Failed to decode JSON from server: {data.decode('utf-8') if data else 'No data'} - {e}")
                        # 오류 발생 시 계속 진행할지, 중단할지 결정
                    except Exception as e: # sendall 등 다른 소켓 오류
                        print(f"[TCP CLIENT ERROR] Communication error: {e}")
                        break
                
                    # 주기적인 작업 후 종료 신호 확인 및 딜레이
                    # time.sleep(1) # 원래 코드의 1초 딜레이
                    for _ in range(10): # 1초 대기 (0.1초 간격으로 종료 신호 확인)
                        if not self.running:
                            break
                        self.msleep(100)
        
            except socket.timeout: # connect 타임아웃
                print(f"[TCP CLIENT ERROR] Connection to {SERVER_IP}:{TCP_SERVER_PORT} timed out.")
            except ConnectionRefusedError:
                print(f"[TCP CLIENT ERROR] Connection refused by server {SERVER_IP}:{TCP_SERVER_PORT}.")
            except Exception as e:
                print(f"[TCP CLIENT ERROR] An unexpected error occurred: {e}")
            finally:
                print("[TCP CLIENT] Connection closed.")

    def stop(self):
        self.running = False
                
# UDP client
class UdpClientThread(QThread):
    frame_from_server = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)  # 기본 웹캠

        if not cap.isOpened():
            print("[UDP CLIENT ERROR] Cannot open webcam.")
            return

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            print("[UDP CLIENT] Started.")
            client_socket.settimeout(0.5)
            #display_window_name = "UDP Client - Video Stream"

            while cap.isOpened() and self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[UDP CLIENT] Failed to grab frame.")
                    break
            
                frame_resized = cv2.resize(frame, (640, 480))
            
                # 인코딩: JPEG 포맷으로 압축
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                ret, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
                if not ret:
                    print("[UDP CLIENT] Failed to encode frame.")
                    #cv2.imshow(display_window_name, frame_resized)
                #if cv2.waitKey(1) & 0xFF == 27: break
                    self.frame_from_server.emit(frame_resized)
                    continue

                current_frame = frame_resized
            
                try:
                    client_socket.sendto(buffer.tobytes(), (SERVER_IP, UDP_SERVER_PORT))

                    try:
                        data_received, server_addr = client_socket.recvfrom(65536) # 충분한 버퍼 크기

                        # 수신된 데이터(바이트)를 numpy 배열로 변환 후 이미지로 디코딩
                        processed_buffer = np.frombuffer(data_received, dtype=np.uint8)
                        processed_frame = cv2.imdecode(processed_buffer, cv2.IMREAD_COLOR)

                        if processed_frame is not None:
                            current_frame = processed_frame # 성공적으로 받으면 처리된 프레임으로 교체
                            # print(f"[UDP CLIENT] Received processed frame from {server_addr}")
                        #else:
                            #print("[UDP CLIENT] Failed to decode processed frame from server.")
                            # 디코딩 실패 시 current_frame_to_display는 여전히 frame_resized (보낸 프레임)

                    except socket.timeout:
                        print("[UDP CLIENT] Timeout waiting for processed frame from server.")
                        # 타임아웃 시 current_frame_to_display는 여전히 frame_resized (보낸 프레임)
                    except Exception as e:
                        print(f"[UDP CLIENT ERROR] Error receiving/decoding processed frame: {e}")
                        # 기타 수신/디코딩 오류 시 current_frame_to_display는 여전히 frame_resized

                except Exception as e:
                    print(f"[UDP CLIENT ERROR] Failed to send data: {e}")
                    break # 전송 실패는 심각한 오류로 간주하고 루프 중단

                self.frame_from_server.emit(current_frame)
                self.msleep(33)

        print("[UDP CLIENT] Releasing resources.")
        cap.release()
        #cv2.destroyAllWindows()
        print("[UDP CLIENT] Stopped.")

    def stop(self):
        self.running = False

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("COVA II")

        self.pixmap = QPixmap()
        self.tcp_client = TcpClientThread()
        self.udp_client = UdpClientThread()
        self.tcp_client.start()
        self.udp_client.start()

        self.tcp_client.msg_from_server.connect(self.update_msg)
        self.udp_client.frame_from_server.connect(self.update_video)

        self.label_msg.setText("Waiting for Message...")

    def update_msg(self, msg):
        self.label_msg.setText(msg)

    def update_video(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, c = rgb_image.shape
        qimage = QImage(rgb_image.data, w, h, w*c, QImage.Format.Format_RGB888)
        
        self.pixmap = self.pixmap.fromImage(qimage)
        self.pixmap = self.pixmap.scaled(self.label_video.width(), self.label_video.height())

        self.label_video.setPixmap(self.pixmap)

# main

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()

    sys.exit(app.exec())

'''
def tcp_client(shutdown_event):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        try:
            client.connect((SERVER_IP, TCP_SERVER_PORT))
            print("[TCP CLIENT] Connected to server.")
            client.settimeout(1.0)  # recv 호출에 대한 타임아웃 설정 (1초)

            message_counter = 0
            while not shutdown_event.is_set():
                message_counter += 1
                # 서버에 JSON 메시지 전송
                message_to_send = {"type": "tcp_request", "id": message_counter, "data": "Hello from TCP client"}
                try:
                    client.sendall(json.dumps(message_to_send).encode('utf-8'))
                    print(f"[TCP CLIENT] Sent: {message_to_send}")

                    # 서버로부터 응답 수신
                    data = client.recv(1024) # 타임아웃 적용
                    if not data:
                        print("[TCP CLIENT] Server closed connection.")
                        break
                    
                    response_message = json.loads(data.decode('utf-8'))
                    print(f"[TCP CLIENT] Received: {response_message}")

                except socket.timeout:
                    # print("[TCP CLIENT] recv timed out. Checking shutdown signal.")
                    continue # 루프를 계속하여 shutdown_event 확인
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                    print(f"[TCP CLIENT ERROR] Connection error: {e}")
                    break
                except json.JSONDecodeError as e:
                    print(f"[TCP CLIENT ERROR] Failed to decode JSON from server: {data.decode('utf-8') if data else 'No data'} - {e}")
                    # 오류 발생 시 계속 진행할지, 중단할지 결정
                except Exception as e: # sendall 등 다른 소켓 오류
                    print(f"[TCP CLIENT ERROR] Communication error: {e}")
                    break
                
                # 주기적인 작업 후 종료 신호 확인 및 딜레이
                # time.sleep(1) # 원래 코드의 1초 딜레이
                for _ in range(10): # 1초 대기 (0.1초 간격으로 종료 신호 확인)
                    if shutdown_event.is_set():
                        break
                    time.sleep(0.1)
        
        except socket.timeout: # connect 타임아웃
            print(f"[TCP CLIENT ERROR] Connection to {SERVER_IP}:{TCP_SERVER_PORT} timed out.")
        except ConnectionRefusedError:
            print(f"[TCP CLIENT ERROR] Connection refused by server {SERVER_IP}:{TCP_SERVER_PORT}.")
        except Exception as e:
            print(f"[TCP CLIENT ERROR] An unexpected error occurred: {e}")
        finally:
            print("[TCP CLIENT] Connection closed.")
                
# UDP client

def udp_client(shutdown_event):
    cap = cv2.VideoCapture(0)  # 기본 웹캠

    if not cap.isOpened():
        print("[UDP CLIENT ERROR] Cannot open webcam.")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
        print("[UDP CLIENT] Started.")
        client_socket.settimeout(0.5)
        display_window_name = "UDP Client - Video Stream"

        while cap.isOpened() and not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[UDP CLIENT] Failed to grab frame.")
                break
            
            frame_resized = cv2.resize(frame, (640, 480))
            
            # 인코딩: JPEG 포맷으로 압축
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ret, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
            if not ret:
                print("[UDP CLIENT] Failed to encode frame.")
                cv2.imshow(display_window_name, frame_resized)
                if cv2.waitKey(1) & 0xFF == 27: break
                continue

            current_frame_to_display = frame_resized
            
            try:
                client_socket.sendto(buffer.tobytes(), (SERVER_IP, UDP_SERVER_PORT))

                try:
                    data_received, server_addr = client_socket.recvfrom(65536) # 충분한 버퍼 크기

                    # 수신된 데이터(바이트)를 numpy 배열로 변환 후 이미지로 디코딩
                    processed_buffer = np.frombuffer(data_received, dtype=np.uint8)
                    processed_frame = cv2.imdecode(processed_buffer, cv2.IMREAD_COLOR)

                    if processed_frame is not None:
                        current_frame_to_display = processed_frame # 성공적으로 받으면 처리된 프레임으로 교체
                        # print(f"[UDP CLIENT] Received processed frame from {server_addr}")
                    else:
                        print("[UDP CLIENT] Failed to decode processed frame from server.")
                        # 디코딩 실패 시 current_frame_to_display는 여전히 frame_resized (보낸 프레임)

                except socket.timeout:
                    print("[UDP CLIENT] Timeout waiting for processed frame from server.")
                    # 타임아웃 시 current_frame_to_display는 여전히 frame_resized (보낸 프레임)
                except Exception as e:
                    print(f"[UDP CLIENT ERROR] Error receiving/decoding processed frame: {e}")
                    # 기타 수신/디코딩 오류 시 current_frame_to_display는 여전히 frame_resized

            except Exception as e:
                print(f"[UDP CLIENT ERROR] Failed to send data: {e}")
                break # 전송 실패는 심각한 오류로 간주하고 루프 중단

            cv2.imshow(display_window_name, current_frame_to_display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키
                print("[UDP CLIENT] ESC key pressed, stopping.")
                break

        print("[UDP CLIENT] Releasing resources.")
        cap.release()
        cv2.destroyAllWindows()
        print("[UDP CLIENT] Stopped.")

if __name__ == "__main__":
    shutdown_event = threading.Event()

    tcp_thread = threading.Thread(target=tcp_client, args=(shutdown_event,), daemon=True)
    udp_thread = threading.Thread(target=udp_client, args=(shutdown_event,), daemon=True)

    print("[MAIN] Starting TCP and UDP client threads...")
    tcp_thread.start()
    udp_thread.start()

    try:
        # 메인 스레드가 데몬 스레드들이 실행되는 동안 살아있도록 유지
        # KeyboardInterrupt (Ctrl+C)를 통해 종료 신호 보냄
        while True:
            if not tcp_thread.is_alive() and not udp_thread.is_alive():
                print("[MAIN] Both threads have finished.")
                break
            if shutdown_event.is_set(): # 다른 스레드에서 shutdown_event를 설정한 경우
                print("[MAIN] Shutdown event detected by main thread.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt received! Signaling threads to shut down...")
        shutdown_event.set()
'''
