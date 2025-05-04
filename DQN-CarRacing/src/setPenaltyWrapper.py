import gymnasium as gym
import numpy as np

class setPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty = -0.25):
        super().__init__(env)
        self.penalty = penalty
        self.cx, self.cy = 48, 72
        self.roi_size = 4

        # ROI 좌표 설정
        self.rois = [
            (self.cx - 4, self.cy),        # 왼쪽 측면
            (self.cx + 3, self.cy),        # 오른쪽 측면
            (self.cx - 4, self.cy - 8),    # 전방 왼쪽
            (self.cx + 3, self.cy - 8),    # 전방 오른쪽
        ]
    #step overide
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        count = 0  # 페널티 대상 ROI 개수

        for (x, y) in self.rois:
            roi = obs[y - self.roi_size//2:y + self.roi_size//2,
                      x - self.roi_size//2:x + self.roi_size//2]

            mean_color = roi.mean(axis=(0, 1))  # RGB 평균

            if not self._is_gray(mean_color):
                count += 1

        reward += self.penalty * count

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _is_gray(self, rgb, thr=20):
        r, g, b = rgb
        return abs(r - g) < thr and abs(g - b) < thr and abs(r - b) < thr


