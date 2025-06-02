import torch
import cv2
import numpy as np
from unet import UNet

def predict_video(model_path, input_video_path, output_video_path, input_size=(512, 256)):
    """
    영상에 대한 세그멘테이션 마스크를 프레임마다 예측하고, 마스크 영상 저장

    Args:
        model_path (str): 학습된 모델의 경로 (.pth)
        input_video_path (str): 입력 영상 경로
        output_video_path (str): 저장할 출력 영상 경로
        input_size (tuple): 모델 입력 사이즈 (width, height)
    """
    # ===== 모델 로딩 =====
    model = UNet(num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ===== 영상 열기 =====
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    print("영상 추론 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 원본 프레임 보존
        original_frame = frame.copy()

        # 전처리
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, input_size)
        img_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # 추론
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.argmax(output, dim=1)
            pred_mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)

        # 마스크 컬러화
        h, w = input_size[1], input_size[0]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[pred_mask_np == 1] = [0, 255, 0]
        color_mask[pred_mask_np == 2] = [0, 0, 255]
        color_mask_resized = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))

        # 마스크와 원본 합성 (alpha blending)
        overlay = cv2.addWeighted(original_frame, 0.7, color_mask_resized, 0.3, 0)

        # 저장
        out.write(overlay)

    cap.release()
    out.release()
    print(f"저장 완료: {output_video_path}")
