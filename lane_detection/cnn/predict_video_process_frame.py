import torch
import cv2
import numpy as np
from unet2 import UNet
from process_frame import process_frame

def predict_video_process_frame(model_path, input_video_path, output_video_path, input_size=(512, 256), show_live=True):
    """
    실시간 키 입력 기반 mode(center/left/right) 전환 지원 영상 추론

    - C: center 모드
    - L: left 모드
    - R: right 모드
    - ESC: 종료
    """
    # 색상 정의: (BGR)
    color_map = {
        0: [0, 0, 0],         # 배경 - 검정
        1: [255, 255, 255],   # 흰색 실선 - 흰색
        2: [128, 128, 128],   # 흰색 점선 - 회색
        3: [0, 255, 255],     # 중앙선(노란 실선) - 노랑
        4: [0, 0, 255],       # 정지선 - 빨강
        5: [0, 255, 0],       # 횡단보도 - 초록
    }

    # 모델 로드
    model = UNet(num_classes=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 비디오 열기
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"입력 영상 열기 실패: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    print("[INFO] 영상 추론 시작 (C=center, L=left, R=right, ESC=종료)")
    uuid_counter = 0
    mode = "center"  # 기본 모드

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 차선 추론 및 조향각 계산
        result = process_frame(
            frame=frame,
            model=model,
            device=device,
            uuid=uuid_counter,
            input_size=input_size,
            mode=mode
        )
        uuid_counter += 1

        overlay = frame.copy()
        angle = result["steering_angle"]

        if angle is not None:
            h, w = overlay.shape[:2]
            center = (w // 2, h)
            length = 100
            rad = np.radians(angle)
            end_point = (
                int(center[0] + length * np.sin(rad)),
                int(center[1] - length * np.cos(rad))
            )
            cv2.arrowedLine(overlay, center, end_point, (0, 255, 0), 3)
            cv2.putText(overlay, f"Angle: {angle:.2f} deg", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "No Steering", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 현재 모드 표시
        cv2.putText(overlay, f"Mode: {mode.upper()}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # 마스크 컬러화
        color_mask = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            color_mask[result.get("pred_mask") == class_id] = color

        # 마스크와 원본 합성 (alpha blending)
        color_mask_resized = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(overlay, 0.7, color_mask_resized, 0.3, 0)

                # --- [1] 인식된 차선 skeleton 시각화 ---
        for x, y in result["skeleton_points"]:
            cv2.circle(overlay, (x, y), 2, (0, 0, 255), -1)  # 빨간 점

        # --- [2] 영상 중앙선 표시 ---
        h, w = overlay.shape[:2]
        center_x = w // 2
        cv2.line(overlay, (center_x, 0), (center_x, h), (255, 255, 255), 2)

        # --- [3] 추출된 차선 중심선 표시 ---
        avg_center_x_mask_res = result.get("avg_center_x_mask_res")
        if avg_center_x_mask_res is not None:
            scale_x = w / result["pred_mask_shape"][1]
            avg_center_x_orig = int(avg_center_x_mask_res * scale_x)
            cv2.line(overlay, (avg_center_x_orig, 0), (avg_center_x_orig, h), (0, 255, 255), 2)
            cv2.putText(overlay, "LaneCenter", (avg_center_x_orig + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- [4] 오프셋 수치 표시 ---
        offset = result.get("offset")
        if offset is not None:
            cv2.putText(overlay, f"Offset: {offset:+.1f}px", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 100), 2)


        out.write(overlay)

        if show_live:
            cv2.imshow("Steering Visualization", overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[INFO] ESC 누름 → 종료")
                break
            elif key == ord('c'):
                mode = "center"
                print("[INPUT] → mode 변경: center")
            elif key == ord('l'):
                mode = "left"
                print("[INPUT] → mode 변경: left")
            elif key == ord('r'):
                mode = "right"
                print("[INPUT] → mode 변경: right")

    cap.release()
    out.release()
    if show_live:
        cv2.destroyAllWindows()

    print(f"[INFO] 저장 완료: {output_video_path}")
