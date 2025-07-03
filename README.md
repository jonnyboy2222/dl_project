# 🚗 K-Urban AutoDrive feat. Don't Crash
**도심형 자율주행 보조를 위한 딥러닝 보조 시스템**  
한국 도심 환경에 특화된 자율주행 시스템의 핵심 기술 개발 프로젝트
---
## 시연 영상
[img](link)
---
## 핵심 특징 요약
- 한국 도심 환경을 반영한 자율주행 시뮬레이션 트랙 제작
- YOLO 기반 딥러닝 모델을 활용한 교통 실시간 객체 인식
- 커스텀 CNN 기반 딥러닝 모델을 활용한 실시간 차선 인식
- 차선 정보 및 경고 메시지를 포함한 실시간 GUI 시각화 제공
- 서버-클라이언트 기반 실시간 영상 송수신 및 추론 결과 전송 구조 구축
- 객체 감지 및 주행 동작 기록을 위한 데이터베이스 구조 설계
- 정지선, 표지판, 교차로 등 다양한 시나리오 기반 테스트 수행
---
## 목차
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Team Information](#team-information)  
4. [Development Environment](#development-environment)  
5. [Design](#design)  
    - [User Requirements](#user-requirements)  
    - [System Requirements](#system-requirements)  
    - [System Architecture](#system-architecture)  
    - [Interface Specification](#interface-specification)  
    - [Data Structure](#data-structure)  
    - [Scenario](#scenario)  
    - [GUI Configuration](#gui-configuration)  
    - [Test Case](#test-case)  
6. [Limitations](#limitations)  
7. [Conclusion and Future Works](#conclusion-and-future-works)

---
## Overview
**K-Urban AutoDrive**는 한국 도심 환경에 최적화된 자율주행 보조 시스템으로, 복잡한 교통 상황에서도 안전하고 효율적인 주행을 가능하게 하기 위한 프로젝트입니다.

본 시스템은 실제 한국 도심을 반영한 시뮬레이션 환경을 구축하고, 딥러닝 기반 객체 인식 및 차선 인식 모델을 통해 실시간 주행 판단을 수행합니다. 운전자의 시야를 보조하는 GUI를 통해 주행 중 감지된 객체와 경고 정보를 시각화하며, 서버-클라이언트 기반 구조를 통해 실시간 데이터 처리와 통신 효율성을 확보하였습니다.

또한 주행 중 감지된 객체, 수행 동작, 운전 회차 기록 등을 데이터베이스에 저장하여 분석 가능성을 높였으며, 다양한 주행 시나리오 기반 테스트를 통해 시스템의 실효성을 검증하였습니다.
---
## Key Features

