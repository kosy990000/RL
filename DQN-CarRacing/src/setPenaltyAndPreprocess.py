import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
    

def preprocess(img):
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img.astype(np.float32)



class setPenaltyAndPreprocess(gym.Wrapper):
    def __init__(            
            self,
            env,
            skip_frames=4,
            stack_frames=4,
            initial_no_op=50,
            penalty = 0.025,
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
            
            r += self.penalty * count

            reward += r

            if terminated or truncated:
                break
            
            # ------ 여기서 바로 ROI 표시 ------
            #self._show_roi_live(s)


        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info
    

    def _is_gray(self, rgb, thr=20):
        r, g, b = rgb
        return abs(r - g) < thr and abs(g - b) < thr and abs(r - b) < thr

    
    def _show_roi_live(self, obs, roi_size=4):
        obs_copy = obs.copy()

        for (x, y) in self.rois:
            x1, y1 = x - roi_size // 2, y - roi_size // 2
            x2, y2 = x + roi_size // 2, y + roi_size // 2

            roi = obs[y1:y2, x1:x2]
            mean = roi.mean(axis=(0, 1))

            color = (0, 255, 0) if self._is_gray(mean) else (0, 0, 255)

            obs_copy = cv2.rectangle(obs_copy, (x1, y1), (x2, y2), color, 1)

        obs_copy = cv2.cvtColor(obs_copy, cv2.COLOR_BGR2RGB)
        plt.imshow(obs_copy)
        plt.title(f"ROI (Step)")
        plt.axis("off")
        plt.pause(0.01)   # <-- 이거 넣어야 실시간 갱신됨
        plt.clf()
