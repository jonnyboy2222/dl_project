import pygame
from env import ParkingEnv

pygame.init()
env = ParkingEnv()
clock = pygame.time.Clock()
obs = env.reset()
done = False
while not done:
    action = [0.0, 0.5]  # 앞으로 가기
    obs, reward, done, info = env.step(action)
    env.render()
    clock.tick(30)  # FPS 제한