import torch
import cv2
import numpy as np
from unet2 import UNet
from skimage.morphology import skeletonize

def predict_webcam(model_path, input_size=(512, 256), save_output=False, output_path="webcam_output.mp4"):
    """
    웹캠 실시간 차선 세그멘테이션 추론 함수

    Args:
        model_path (str): 학습된 모델의 경로 (.pth)
        input_size (tuple): 모델 입력 사이즈 (width, height)
        save_output (bool): 웹캠 영상 저장 여부
        output_path (str): 저장할 영상 경로
    """
    # ===== 모델 로딩 =====
    model = UNet(num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ===== 웹캠 열기 =====
    cap = cv2.VideoCapture(0)  # 기본 카메라
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다")

    # 저장 옵션 (선택)
    if save_output:
        fps = 20.0  # 웹캠에 따라 적절히 설정
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("웹캠 추론 시작 (ESC 키로 종료)...")

    # 초기화
    prev_angle = 0.0
    prev_center_x = None
    max_x_change = 3
    alpha = 0.05               # EMA 스무딩 계수
    delta_limit = 0.5          # 프레임 간 최대 조향각 변화 (degrees)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break

        original_frame = frame.copy()

        # 전처리
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, input_size)
        img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)



        # --- Skeleton 기반 중앙선 추정 ---
        h, w = pred_mask.shape
        frame_center_x = w // 2
        avg_center_x = None
        angle_deg = None
        
        # 좌우 차선 (class 1, 2) 병합 → 중심선 생성
        center_mask = np.logical_or(pred_mask == 1, pred_mask == 2)
        skeleton = skeletonize(center_mask)

        center_points = np.column_stack(np.where(skeleton))  # [y, x]

        if len(center_points) > 10:
            # 하단 1/3만 고려 (하단에서 주행 판단이 유효)
            lower_third = center_points[center_points[:, 0] > h * 0.66]
            if len(lower_third) > 0:
                avg_center_x = int(np.mean(lower_third[:, 1]))

                # --- 중앙선 x 변화 제한 ---
                if prev_center_x is not None:
                    dx = avg_center_x - prev_center_x
                    if abs(dx) > max_x_change:
                        avg_center_x = prev_center_x + np.sign(dx) * max_x_change
                prev_center_x = avg_center_x

                offset_x = avg_center_x - frame_center_x
                angle_rad = np.arctan2(offset_x, h // 2)
                angle_deg = np.degrees(angle_rad)

                # --- EMA + 변화량 제한 ---
                raw_angle = (1 - alpha) * prev_angle + alpha * angle_deg
                delta = raw_angle - prev_angle
                if abs(delta) > delta_limit:
                    delta = np.sign(delta) * delta_limit
                smoothed_angle = prev_angle + delta
                prev_angle = smoothed_angle

                prev_angle = smoothed_angle

                print(f"[조향각] 원각: {angle_deg:.2f}, EMA+제한: {smoothed_angle:.2f}")
            else:
                print("중심선이 하단에서 감지되지 않음")
        else:
            print("중심선 추출 실패")

        # 시각화
        in_h, in_w = input_size[1], input_size[0]
        color_mask = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        color_mask[pred_mask == 1] = [0, 255, 0]     # 좌차선 - 초록
        color_mask[pred_mask == 2] = [0, 0, 255]     # 우차선 - 빨강
        color_mask[skeleton] = [255, 255, 0]         # 중앙선 - 노랑

        color_mask_resized = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(original_frame, 0.7, color_mask_resized, 0.3, 0)

        # 중앙선 및 프레임 중심 표시
        if avg_center_x is not None:
            scale_w = frame.shape[1] / w
            scale_h = frame.shape[0] / h
            cx = int(avg_center_x * scale_w)
            cy_bottom = int(h * scale_h)
            cy_top = int(h * 0.5 * scale_h)
            cv2.line(overlay, (cx, cy_bottom), (cx, cy_top), (255, 255, 0), 2)
        cv2.line(overlay, (frame.shape[1]//2, frame.shape[0]), (frame.shape[1]//2, int(frame.shape[0]*0.5)), (0, 255, 255), 1)

        cv2.imshow("Webcam Segmentation", overlay)

        if save_output:
            out.write(overlay)

        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC 누름 → 종료")
            break

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

predict_webcam("best_model_final.pth")