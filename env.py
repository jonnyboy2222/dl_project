import gym
from gym import spaces
import numpy as np
import pygame
import random

class ParkingEnv(gym.Env):
    def __init__(self,
                 screen_size=(800, 600),
                 parked_slots=None,
                 agent_vehicle_type="suv",
                 parked_vehicle_types=None,
                 agent_start_distance=200):
        super().__init__()
        # 시뮬레이션 설정
        self.width, self.height = screen_size
        self.surface = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Autonomous Parking")
        # 차량 속성 설정
        self.vehicle_types = {
            "truck": (70, 40),
            "suv": (60, 35),
            "sedan": (55, 30),
            "compact": (45, 25)
        }
        self.agent_vehicle_type = agent_vehicle_type
        self.agent_size = self.vehicle_types[agent_vehicle_type]
        self.parked_slots = parked_slots or [False] * 6
        self.parked_vehicle_types = parked_vehicle_types or ["sedan"] * 6
        self.agent_start_distance = agent_start_distance
        # 로그 저장
        self.logs = []
        # 상태/행동 공간 설정
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.reset()

    def reset(self):
        self.agent_pos = [self.width // 2, self.agent_start_distance]
        self.agent_angle = 0
        self.agent_velocity = 0
        self.logs = []
        return self._get_state()
    
    def step(self, action):
        steer, throttle = action
        self.agent_angle += steer * 5
        self.agent_velocity = throttle * 5
        self.agent_pos[0] += self.agent_velocity * np.cos(np.radians(self.agent_angle))
        self.agent_pos[1] += self.agent_velocity * np.sin(np.radians(self.agent_angle))
        # reward/done 임시
        reward = -0.1
        done = False
        # 로그 저장
        self.logs.append({
            "pos": tuple(self.agent_pos),
            "angle": self.agent_angle,
            "action": action,
            "reward": reward
        })
        return self._get_state(), reward, done, {}
    
    def render(self, mode="human"):
        self.surface.fill((230, 230, 230))  # 배경색
        # 주차라인 그리기
        for i in range(3):
            pygame.draw.rect(self.surface, (180, 180, 180), (50, 100 + i*120, 80, 100))  # 왼쪽
            pygame.draw.rect(self.surface, (180, 180, 180), (self.width - 130, 100 + i*120, 80, 100))  # 오른쪽
        # 에이전트 그리기
        rect = pygame.Rect(0, 0, *self.agent_size)
        rect.center = self.agent_pos
        rotated = pygame.transform.rotate(pygame.Surface(self.agent_size), -self.agent_angle)
        rotated.fill((0, 128, 255))
        self.surface.blit(rotated, rotated.get_rect(center=rect.center))
        # 정보 출력
        font = pygame.font.SysFont("Arial", 16)
        info = f"Pos: {tuple(round(x, 1) for x in self.agent_pos)}  | Angle: {round(self.agent_angle, 1)}°"
        info += f"  | Velocity: {round(self.agent_velocity, 1)}"
        text = font.render(info, True, (0, 0, 0))
        self.surface.blit(text, (10, 10))
        pygame.display.flip()

    def _get_state(self):
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.agent_angle,
            self.agent_velocity,
            0, 0  # 추후: 거리, 정렬도 등 추가 가능
        ], dtype=np.float32)
    

