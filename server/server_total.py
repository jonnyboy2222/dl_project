import socket
import struct
import threading
import json
import numpy as np
import cv2
import time
from process_frame import process_frame
from unet2 import UNet
import torch

TCP_SERVER_IP = '0.0.0.0'
TCP_SERVER_PORT = 12345

UDP_SERVER_IP = '0.0.0.0'
UDP_SERVER_PORT = 54321

# 전역 모델 초기화
model = UNet(num_classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/home/john/dev_ws/dl_project/best_model_final.pth", map_location=device))
model = model.to(device)
model.eval()

def handle_client(conn, addr):
    print(f"[PC2] Connected from {addr}")
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break

            result_msg = {"message": "sample"}
            conn.sendall(json.dumps(result_msg).encode("utf-8"))
            print(f"[PC2] Sent verification result: {result_msg}")
            time.sleep(1)

    except Exception as e:
        print(f"[PC2 ERROR] {e}")
    finally:
        conn.close()
        print(f"[PC2] Disconnected from {addr}")


def udp_video_receiver():
    udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server.bind((UDP_SERVER_IP, UDP_SERVER_PORT))
    print(f"[PC2] UDP Server listening on {UDP_SERVER_IP}:{UDP_SERVER_PORT}")
    
    fps_limit = 15  # 제한 프레임 속도
    frame_interval = 1.0 / fps_limit
    prev_time = 0

    try:
        while True:
            data, addr = udp_server.recvfrom(65535)
            np_data = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            current_time = time.time()
            if current_time - prev_time < frame_interval:
                continue  # 프레임 속도 제한
            prev_time = current_time

            # 세그멘테이션 처리
            overlay = process_frame(frame, model, device)

            # 인코딩 및 전송
            ret, encoded = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                udp_server.sendto(encoded.tobytes(), addr)

            # 디버깅용 표시
            cv2.imshow("Server Processed Frame", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        print(f"[UDP ERROR] {e}")
    finally:
        udp_server.close()
        cv2.destroyAllWindows()



def main():
    # UDP 수신 스레드 먼저 실행
    threading.Thread(target=udp_video_receiver, daemon=True).start()

    # TCP 서버 시작
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((TCP_SERVER_IP, TCP_SERVER_PORT))
        server.listen()
        print(f"[PC2] TCP Server listening on {TCP_SERVER_IP}:{TCP_SERVER_PORT}")

        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()

