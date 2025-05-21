import pygame
import math
import numpy as np
import random

# 환경 파라미터
WIDTH, HEIGHT = 800, 600
FPS = 30

# 색상
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (50, 150, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = -90  # 방향 각도
        self.speed = 0
        self.length = 100
        self.width = 55

    def update(self, action):
        # steer, throttle = action  # -1~1
        steer, throttle_input = action  # steer: -1 (좌) ~ 1 (우), throttle_input: -1 (후진) ~ 1 (전진)
        # self.angle += steer * 5  # 회전
        # self.speed += throttle * 0.5
        # 가속 및 감속 로직
        acceleration_rate = 0.5  # 가속/후진 시 속도 변화량
        deceleration_rate = 0.1  # 마찰 또는 키를 뗐을 때 감속되는 속도 변화량

        if throttle_input != 0:
            # 위/아래 방향키 입력이 있으면 해당 방향으로 가속
            self.speed += throttle_input * acceleration_rate
        else:
            # 위/아래 방향키 입력이 없으면 서서히 감속
            if self.speed > deceleration_rate:
                self.speed -= deceleration_rate
            elif self.speed < -deceleration_rate:
                self.speed += deceleration_rate
            else:
                self.speed = 0  # 속도가 매우 낮으면 완전히 정지

        # 조향 로직: 속도 방향에 따라 조향 효과 반전
        # 전진 (speed > 0): steer > 0 -> angle 증가 (시계 방향)
        # 후진 (speed < 0): steer > 0 -> angle 감소 (반시계 방향)
        if self.speed > 0:
            self.angle += steer * 5
        elif self.speed < 0:
            self.angle -= steer * 5
        
        # speed == 0 일 때는 각도 변화 없음 (정지 상태 조향은 이 모델에서 고려 안 함)

        self.speed = np.clip(self.speed, -3, 3)

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

        self.angle %= 360 # Keep angle between 0 and 360
        if self.angle < 0: # Handle negative results from modulo
            self.angle += 360

    def draw(self, screen):
        rad = math.radians(self.angle)
        # 1. 원본 Surface 생성 (길이 x 폭)
        original_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)  # 투명 배경
        # 2. 자동차 몸체 색상 채우기
        original_surface.fill((0, 0, 0))
        # 2.1. 자동차 헤드라이트 그리기
        # 헤드라이트 크기 및 위치 설정
        headlight_depth = int(self.length * 0.08)  # 헤드라이트의 깊이 (자동차 길이 방향)
        headlight_span = int(self.width * 0.25)   # 각 헤드라이트의 폭 (자동차 폭 방향)
        headlight_side_margin = int(self.width * 0.15) # 자동차 가장자리로부터 헤드라이트까지의 간격

        headlight_x_pos = self.length - headlight_depth # 헤드라이트의 x 좌표 (자동차 앞쪽 끝)

        # 위쪽 헤드라이트
        pygame.draw.rect(original_surface, (255,255,0), (headlight_x_pos, headlight_side_margin, headlight_depth, headlight_span))
        # 아래쪽 헤드라이트
        pygame.draw.rect(original_surface, (255,255,0), (headlight_x_pos, self.width - headlight_side_margin - headlight_span, headlight_depth, headlight_span))
        # 3. 회전
        rotated_surface = pygame.transform.rotate(original_surface, -self.angle)
        # 4. 중심을 유지한 채로 회전된 이미지의 위치 계산
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))
        # 5. 화면에 그리기
        screen.blit(rotated_surface, rotated_rect)

    def get_rect(self):
        """Returns the axis-aligned bounding box of the rotated car."""
        # Create a temporary surface with the car's dimensions
        temp_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        rotated_surface = pygame.transform.rotate(temp_surface, -self.angle)
        return rotated_surface.get_rect(center=(self.x, self.y))

    def get_state(self):
        return np.array([self.x / WIDTH, self.y / HEIGHT, self.angle / 360, self.speed / 3])

class Environment:
    def __init__(self):
        # self.car = Car(100, 60)
        # self.goal = pygame.Rect(screen.get_width() - 130, 100 + 2*120, 110, 70)  # 주차 목표 or 출구 방향

        # self.obstacles1 = [pygame.Rect(50,100 + 2*120, 110, 70)]
        # self.obstacles2 = [pygame.Rect(screen.get_width() - 130, 100 + 1*120, 110, 70)]
        # self.obstacles3 = [pygame.Rect(50,100 + 0*120, 110, 70)]

        # self.obstacles = [self.obstacles1, self.obstacles2, self.obstacles3]

        # self.obs_list = []
        
        # for i in range(3):

        #     self.obstacles = [pygame.Rect(x, y, 110, 70)]

        #     # 50, 100 + i*120
        #     # self.screen.get_width() - 130, 100 + i*120

        # 잠재적인 주차 슬롯 위치 정의 (왼쪽 3개, 오른쪽 3개)
        self.potential_slots = []
        for i in range(3):
            # 왼쪽 슬롯
            self.potential_slots.append(pygame.Rect(50, 100 + i*120, 110, 70))
            # 오른쪽 슬롯
            self.potential_slots.append(pygame.Rect(WIDTH - 130, 100 + i*120, 110, 70))

        self.goal = None
        self.all_obstacles = []

    def reset(self):
        self.car = Car(400, 550)
        # 잠재적인 슬롯 위치를 섞어서 랜덤하게 선택
        shuffled_slots = random.sample(self.potential_slots, len(self.potential_slots))
        self.goal = shuffled_slots[0] # 첫 번째 슬롯을 목표로 설정
        self.all_obstacles = shuffled_slots[1:4] # 다음 3개 슬롯을 장애물로 설정
        return self.car.get_state()

    def step(self, action):
        self.car.update(action)

        done = False
        reward = -0.01  # 기본 페널티
        car_rect = pygame.Rect(self.car.x, self.car.y, 100, 60)

        # if car_rect.colliderect(self.goal):
        car_bounding_rect = self.car.get_rect() # Get the car's AABB

        # Check for goal completion: car must be fully inside the goal area
        if self.goal.contains(car_bounding_rect):
            reward = 10
            done = True

        # for obs in self.obstacles1:
        #     if car_rect.colliderect(obs):
        else:
            # Check for obstacle collision: any part of the car touches an obstacle
            for obs in self.all_obstacles:
                if car_bounding_rect.colliderect(obs):
                    reward = -10
                    done = True
                    break # Exit loop once a collision is found

        # 경계 충돌 체크 (화면 밖으로 나가는 경우)
        if not pygame.Rect(0, 0, WIDTH, HEIGHT).contains(car_bounding_rect):
                reward = -10
                done = True
        
        # for obs in self.obstacles2:
        #     if car_rect.colliderect(obs):
        #         reward = -10
        #         done = True

        # for obs in self.obstacles3:
        #     if car_rect.colliderect(obs):
        #         reward = -10
        #         done = True

        return self.car.get_state(), reward, done

    def render(self, screen):
        screen.fill((230, 230, 230))
        
        # for i in range(3):
        #     pygame.draw.rect(screen, (180, 180, 180), (50, 100 + i*120, 110, 70))  # 왼쪽
        #     pygame.draw.rect(screen, (180, 180, 180), (screen.get_width() - 130, 100 + i*120, 110, 70))  # 오른쪽

        # 모든 잠재적 주차 슬롯 라인 그리기
        for slot in self.potential_slots:
             pygame.draw.rect(screen, GRAY, slot)

        # 랜덤하게 선택된 목표 및 장애물 그리기

        pygame.draw.rect(screen, GREEN, self.goal)

        # for obs in self.obstacles1:
        for obs in self.all_obstacles:
            pygame.draw.rect(screen, RED, obs)

        # for obs in self.obstacles2:
        #     pygame.draw.rect(screen, RED, obs)

        # for obs in self.obstacles3:
        #     pygame.draw.rect(screen, RED, obs)

        self.car.draw(screen)
        pygame.display.flip()

# 디버깅용 실행 코드
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = Environment()
    state = env.reset()

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 키보드 조작 테스트용
        keys = pygame.key.get_pressed()
        steer = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        throttle = (keys[pygame.K_UP] - keys[pygame.K_DOWN])

        action = [steer, throttle]
        state, reward, done = env.step(action)
        env.render(screen)

        if done:
            print("Episode finished. Reward:", reward)
            state = env.reset()

    pygame.quit()
    exit()