from predict_and_save_mask import predict_and_save_mask
from predict_video import predict_video
import os
import torch
from unet import UNet

# # 샘플 이미지 경로 (이미 다운로드 받은 파일)
# sample_image = "./data/sample.jpg"

# # 학습된 모델 경로
# model_path = "best_model.pth"  # 학습 완료 후 저장한 모델

# # 출력 마스크 저장 경로
# output_mask = "./inference_results/sample_mask.png"

# # 추론 실행
# predict_and_save_mask(model_path, sample_image, output_mask)

# 비디오 경로
sample_video = "./data/sample_video3.mp4"

# 학습된 모델 경로
model_path = "best_model_epoch5.pth"

# 출력 비디오 저장 경로
output_video_path = "./inference_results/sample_video_mask3.mp4"

# 추론 실행
predict_video(model_path, sample_video, output_video_path)
