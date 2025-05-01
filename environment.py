import gym
from gym import spaces
import numpy as np
import cv2
import random

class UAVVictimSearchEnv(gym.Env):
    def __init__(self):
        super(UAVVictimSearchEnv, self).__init__()
        self.grid_size = 15
        self.num_cells = self.grid_size * self.grid_size

        # Two UAVs with discrete actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))

        # Observation is simply the visited grid (for now)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.uint8)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.visited = np.zeros_like(self.grid)

        # Place victims in 10% of the cells
        num_victim_cells = self.num_cells // 15
        self.victim_positions = {}
        for _ in range(num_victim_cells):
            x, y = random.randint(0,  self.grid_size-1), random.randint(0,  self.grid_size-1)
            self.victim_positions[(x, y)] = random.randint(1, 10)

        # UAVs start at opposite corners
        self.uav1_pos = [0, 0]
        self.uav2_pos = [ self.grid_size-1,  self.grid_size-1]

        return self._get_obs()

    def step(self, actions):
        self._move_uav(self.uav1_pos, actions[0])
        self._move_uav(self.uav2_pos, actions[1])

        self.visited[tuple(self.uav1_pos)] = 1
        self.visited[tuple(self.uav2_pos)] = 1

        reward = 0
        for pos in [tuple(self.uav1_pos), tuple(self.uav2_pos)]:
            if pos in self.victim_positions:
                reward += self.victim_positions[pos]
                del self.victim_positions[pos]

        done = len(self.victim_positions) == 0
        return self._get_obs(), reward, done, {}

    def _move_uav(self, pos, action):
        if action == 0 and pos[0] > 0: pos[0] -= 1  # UP
        elif action == 1 and pos[0] < self.grid_size - 1: pos[0] += 1  # DOWN
        elif action == 2 and pos[1] > 0: pos[1] -= 1  # LEFT
        elif action == 3 and pos[1] < self.grid_size - 1: pos[1] += 1  # RIGHT

    def _get_obs(self):
        return np.copy(self.visited)

    def render(self, mode="human"):
        scale = 50     # Each grid cell = 5x5 pixels
        img_size = self.grid_size * scale
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # white background

        # Victims as shades of red
        for (x, y), count in self.victim_positions.items():
            intensity = int(25 + (count / 10) * 230)  # red shades
            cv2.rectangle(
                img,
                (y * scale, x * scale),
                ((y + 1) * scale, (x + 1) * scale),
                (0, 0, intensity),  # BGR: red component
                -1
            )

        # Visited cells as light gray
        visited_coords = np.argwhere(self.visited == 1)
        for x, y in visited_coords:
            cv2.rectangle(
                img,
                (y * scale, x * scale),
                ((y + 1) * scale, (x + 1) * scale),
                (220, 220, 220),
                -1
            )

        # Draw grid lines
        for i in range(self.grid_size + 1):
            cv2.line(img, (0, i * scale), (img_size, i * scale), (180, 180, 180), 1)
            cv2.line(img, (i * scale, 0), (i * scale, img_size), (180, 180, 180), 1)

        # UAV1: Green triangle
        self._draw_triangle(img, self.uav1_pos, scale, (0, 255, 0))

        # UAV2: Blue triangle
        self._draw_triangle(img, self.uav2_pos, scale, (255, 0, 0))

        if mode == "human":
            cv2.imshow("UAV Victim Search", img)
            cv2.waitKey(50)

    def _draw_triangle(self, img, pos, scale, color):
        x, y = pos
        cx, cy = y * scale + scale // 2, x * scale + scale // 2
        points = np.array([
            (cx, cy - scale // 2),
            (cx - scale // 2, cy + scale // 2),
            (cx + scale // 2, cy + scale // 2)
        ])
        cv2.drawContours(img, [points], 0, color, -1)

    def close(self):
        cv2.destroyAllWindows()

# Run the environment
if __name__ == "__main__":
    env = UAVVictimSearchEnv()
    obs = env.reset()
    done = False
    while not done:
        actions = (random.randint(0, 3), random.randint(0, 3))
        obs, reward, done, _ = env.step(actions)
        env.render()
    env.close()
