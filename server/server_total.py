import socket
import struct
import threading
import json
import numpy as np
import cv2
import time

TCP_SERVER_IP = '0.0.0.0'
TCP_SERVER_PORT = 12345

UDP_SERVER_IP = '0.0.0.0'
UDP_SERVER_PORT = 54321


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
    
    try:
        while True:
            try:
                data, addr = udp_server.recvfrom(65535)
                np_data = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

                if frame is not None:
                    cv2.putText(frame, "From Server", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)

                    cv2.imshow("Send Video", frame)

                    # 다시 JPEG로 인코딩
                    ret, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        udp_server.sendto(encoded.tobytes(), addr)  # 다시 클라이언트로 전송

                    if cv2.waitKey(1) == 27:  # ESC
                        break
            except KeyboardInterrupt:
                print("[PC2] KeyboardInterrupt detected. Exiting...")
                break
            except Exception as e:
                print(f"[UDP ERROR] {e}")
                continue

    finally:
        print("[PC2] Closing UDP server and destroying windows.")
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

