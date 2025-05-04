import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
    

def preprocess(img):
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img.astype(np.float32)



class setPenaltyAndPreprocess(gym.RewardWrapper):
    def __init__(            
            self,
            env,
            skip_frames=4,
            stack_frames=4,
            initial_no_op=50,
            penalty = 0.1,
            **kwargs):
        super(setPenaltyAndPreprocess, self).__init__(env, **kwargs)
        
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

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

    def reset(self):
        # Reset the original environment
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info
    
    #step overide
    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            
            # ----- 리워드 수정 -----

            count = 0
            for (x, y) in self.rois:
                roi = s[y-2:y+2, x-2:x+2]
                mean = roi.mean(axis=(0,1))
                if self._is_gray(mean):
                    count += 1

            
            show_roi(s, rois)
            r += self.penalty * count

            reward += r

            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info
    

    def _is_gray(self, rgb, thr=20):
        r, g, b = rgb
        return abs(r - g) < thr and abs(g - b) < thr and abs(r - b) < thr

    
    def show_roi(self, obs, rois, roi_size=4):
        # obs는 (96, 96, 3) 이어야 함 → 이걸 그대로 복사
        obs_copy = obs.copy()
    
        # ROI 그리기
        for (x, y) in rois:
            x1, y1 = x - roi_size // 2, y - roi_size // 2
            x2, y2 = x + roi_size // 2, y + roi_size // 2
    
            # 빨간 사각형 그리기
            obs_copy = cv2.rectangle(obs_copy, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
        # RGB로 변환 (cv2는 BGR임 → matplotlib은 RGB 필요)
        obs_copy = cv2.cvtColor(obs_copy, cv2.COLOR_BGR2RGB)
    
        # 표시
        plt.imshow(obs_copy)
        plt.title("ROI Area Visualization")
        plt.axis("off")
        plt.show()

