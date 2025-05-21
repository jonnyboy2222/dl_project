import pygame
import math
import numpy as np

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
        self.angle = 0  # 방향 각도
        self.speed = 0
        self.length = 40
        self.width = 20

    def update(self, action):
        steer, throttle = action  # -1~1
        self.angle += steer * 5  # 회전
        self.speed += throttle * 0.5
        self.speed = np.clip(self.speed, -3, 3)

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

    def draw(self, screen):
        rad = math.radians(self.angle)
        rotated = pygame.transform.rotate(pygame.Surface((self.length, self.width)), -self.angle)
        rotated.fill(BLUE)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect)

    def get_state(self):
        return np.array([self.x / WIDTH, self.y / HEIGHT, self.angle / 360, self.speed / 3])

class Environment:
    def __init__(self):
        self.car = Car(100, 500)
        self.goal = pygame.Rect(600, 100, 60, 30)  # 주차 목표 or 출구 방향
        self.obstacles = [pygame.Rect(300, 300, 100, 20)]

    def reset(self):
        self.car = Car(100, 500)
        return self.car.get_state()

    def step(self, action):
        self.car.update(action)

        done = False
        reward = -0.01  # 기본 페널티
        car_rect = pygame.Rect(self.car.x-20, self.car.y-10, 40, 20)

        if car_rect.colliderect(self.goal):
            reward = 10
            done = True

        for obs in self.obstacles:
            if car_rect.colliderect(obs):
                reward = -10
                done = True

        return self.car.get_state(), reward, done

    def render(self, screen):
        screen.fill(WHITE)
        pygame.draw.rect(screen, GREEN, self.goal)
        for obs in self.obstacles:
            pygame.draw.rect(screen, RED, obs)
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