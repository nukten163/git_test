import cv2
import mediapipe as mp
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import time
import numpy as np
import collections
import pygame
import os

# Initialize pygame mixer
pygame.mixer.init()

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class GestureApp(ctk.CTk):
    def __init__(self, window_title):
        super().__init__()
        self.title(window_title)
        
        # Windowed mode
        self.geometry("1100x700")
        
        # Configure grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<q>", lambda e: self.destroy())
        
        # Load the emojis and sounds
        self.emoji_path = os.path.join("assets", "emoji.png")
        self.sound_path = os.path.join("assets", "bruh.mp3")
        
        self.wolf_emoji_path = os.path.join("assets", "wolf.png")
        self.wolf_sound_path = os.path.join("assets", "wolf_sound.mp3")
        
        try:
            self.emoji_img = cv2.imread(self.emoji_path, cv2.IMREAD_UNCHANGED)
            self.wolf_emoji_img = cv2.imread(self.wolf_emoji_path, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print("Could not load emojis:", e)
            self.emoji_img = None
            self.wolf_emoji_img = None
            
        # State variables
        self.is_active = False
        self.current_gesture = None
        self.last_gesture = None
        self.gesture_last_seen = 0
        self.grace_period = 0.5 # 0.5s grace period to stop flickering

        # Hand tracking history for 6-7 gesture
        self.history_len = 10
        self.left_y_history = collections.deque(maxlen=self.history_len)
        self.right_y_history = collections.deque(maxlen=self.history_len)

        # MediaPipe Tasks initialization
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Hand Landmarker
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join('assets', 'hand_landmarker.task')),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.hands = HandLandmarker.create_from_options(hand_options)

        # Face Detector
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        face_options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=os.path.join('assets', 'face_detector.task')),
            running_mode=VisionRunningMode.IMAGE
        )
        self.face_detection = FaceDetector.create_from_options(face_options)
        
        # Open video source
        self.vid = cv2.VideoCapture(0)
        
        # Create UI
        self.setup_ui()
        
        self.delay = 15
        self.update_frame()
        self.mainloop()
        
    def setup_ui(self):
        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Gesture Tracker", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(40, 20))
        
        self.btn_activate = ctk.CTkButton(self.sidebar_frame, text="Activate", command=self.toggle_activate,
                                          height=40, font=ctk.CTkFont(size=15, weight="bold"))
        self.btn_activate.grid(row=1, column=0, padx=20, pady=10)
        
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Inactive", text_color="gray")
        self.status_label.grid(row=2, column=0, padx=20, pady=10)
        
        # Legend
        self.legend_label = ctk.CTkLabel(self.sidebar_frame, text="Gestures:\n- 6-7 (Weighing)\n- Wolf (2 Pointers Up)", text_color="lightgray", justify="left")
        self.legend_label.grid(row=3, column=0, padx=20, pady=20)
        
        self.btn_quit = ctk.CTkButton(self.sidebar_frame, text="Quit App", command=self.destroy,
                                      fg_color="#C62828", hover_color="#B71C1C", height=40)
        self.btn_quit.grid(row=5, column=0, padx=20, pady=20)
        
        # Video Frame
        self.video_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="black")
        self.video_frame.grid(row=0, column=1, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
    def toggle_activate(self):
        self.is_active = not self.is_active
        if self.is_active:
            self.btn_activate.configure(text="Deactivate", fg_color="#C62828", hover_color="#B71C1C")
            self.status_label.configure(text="Status: Active", text_color="#4CAF50")
        else:
            self.btn_activate.configure(text="Activate", fg_color=["#3B8ED0", "#1F6AA5"], hover_color=["#36719F", "#144870"])
            self.status_label.configure(text="Status: Inactive", text_color="gray")
            self.current_gesture = None
            self.last_gesture = None
            pygame.mixer.music.stop()
            
    def overlay_image_alpha(self, img, img_overlay, x, y):
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
        
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return
            
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        
        alpha = img_overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        
        for c in range(0, 3):
            img_crop[:, :, c] = (alpha * img_overlay_crop[:, :, c] + alpha_inv * img_crop[:, :, c])
            
    def check_6_7_gesture(self):
        if len(self.left_y_history) < 3 or len(self.right_y_history) < 3:
            return False
            
        left_dy = self.left_y_history[-1] - self.left_y_history[0]
        right_dy = self.right_y_history[-1] - self.right_y_history[0]
        
        threshold = 0.04
        if (left_dy > threshold and right_dy < -threshold) or (left_dy < -threshold and right_dy > threshold):
            return True
        return False
        
    def check_wolf_gesture(self, hand_landmarks_list):
        if not hand_landmarks_list or len(hand_landmarks_list) < 2:
            return False
            
        for landmarks in hand_landmarks_list[:2]:
            # MediaPipe landmarks: 8=Index tip, 6=Index pip, 12=Middle tip, 10=Middle pip
            # 16=Ring tip, 14=Ring pip, 20=Pinky tip, 18=Pinky pip
            index_tip_y = landmarks[8].y
            index_pip_y = landmarks[6].y
            
            middle_tip_y = landmarks[12].y
            middle_pip_y = landmarks[10].y
            
            ring_tip_y = landmarks[16].y
            ring_pip_y = landmarks[14].y
            
            pinky_tip_y = landmarks[20].y
            pinky_pip_y = landmarks[18].y
            
            # Index must be UP (y is smaller)
            if index_tip_y > index_pip_y:
                return False
                
            # Middle, Ring, Pinky must be DOWN (y is larger than pip/mcp)
            if middle_tip_y < middle_pip_y or ring_tip_y < ring_pip_y or pinky_tip_y < pinky_pip_y:
                return False
                
        return True

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            if self.is_active:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # Hand Tracking
                hand_result = self.hands.detect(mp_image)
                
                left_y = None
                right_y = None
                
                if hand_result.hand_landmarks:
                    for landmarks in hand_result.hand_landmarks:
                        x_min = int(min([lm.x for lm in landmarks]) * w)
                        x_max = int(max([lm.x for lm in landmarks]) * w)
                        y_min = int(min([lm.y for lm in landmarks]) * h)
                        y_max = int(max([lm.y for lm in landmarks]) * h)
                        
                        cv2.rectangle(frame_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame_rgb, f"Palm: {x_min},{y_min}", (x_min, max(0, y_min - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    if len(hand_result.hand_landmarks) >= 2:
                        hands_by_x = sorted(hand_result.hand_landmarks[:2], key=lambda h: h[0].x)
                        left_y = hands_by_x[0][0].y
                        right_y = hands_by_x[1][0].y
                            
                # 6-7 Check
                if left_y is not None and right_y is not None:
                    self.left_y_history.append(left_y)
                    self.right_y_history.append(right_y)
                else:
                    self.left_y_history.clear()
                    self.right_y_history.clear()
                    
                is_6_7 = self.check_6_7_gesture()
                
                # Wolf Check
                is_wolf = self.check_wolf_gesture(hand_result.hand_landmarks)
                
                # Update State Machine
                if is_6_7:
                    self.current_gesture = "6_7"
                    self.gesture_last_seen = time.time()
                elif is_wolf:
                    self.current_gesture = "wolf"
                    self.gesture_last_seen = time.time()
                    
                # Grace period for stopping
                if time.time() - self.gesture_last_seen > self.grace_period:
                    self.current_gesture = None
                    
                # Audio State Machine
                if self.current_gesture != self.last_gesture:
                    pygame.mixer.music.stop()
                    
                    if self.current_gesture == "6_7":
                        if os.path.exists(self.sound_path):
                            pygame.mixer.music.load(self.sound_path)
                            pygame.mixer.music.play(-1)
                    elif self.current_gesture == "wolf":
                        if os.path.exists(self.wolf_sound_path):
                            pygame.mixer.music.load(self.wolf_sound_path)
                            pygame.mixer.music.play(-1)
                            
                    self.last_gesture = self.current_gesture
                    
                # Face Detection & Emoji
                if self.current_gesture is not None:
                    face_result = self.face_detection.detect(mp_image)
                    active_emoji = self.emoji_img if self.current_gesture == "6_7" else self.wolf_emoji_img
                    
                    if face_result.detections and active_emoji is not None:
                        detection = face_result.detections[0]
                        bbox = detection.bounding_box
                        
                        orig_x, orig_y = bbox.origin_x, bbox.origin_y
                        orig_w, orig_h = bbox.width, bbox.height
                        
                        cv2.rectangle(frame_rgb, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (0, 255, 0), 2)
                        cv2.putText(frame_rgb, f"Face: {orig_x},{orig_y}", (orig_x, max(0, orig_y - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        bx = orig_x
                        by = orig_y
                        bw = orig_w
                        bh = orig_h
                        
                        pad = int(bw * 0.2)
                        bx -= pad
                        by -= pad
                        bw += pad * 2
                        bh += pad * 2
                        
                        if bw > 0 and bh > 0:
                            try:
                                resized_emoji = cv2.resize(active_emoji, (bw, bh))
                                emoji_rgba = cv2.cvtColor(resized_emoji, cv2.COLOR_BGRA2RGBA)
                                self.overlay_image_alpha(frame_rgb, emoji_rgba, bx, by)
                                
                                text = "six-seven" if self.current_gesture == "6_7" else "куда несет дым"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1
                                thickness = 2
                                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                                text_x = bx + (bw - text_size[0]) // 2
                                text_y = max(0, by - 10)
                                
                                cv2.putText(frame_rgb, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                                cv2.putText(frame_rgb, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                            except Exception as e:
                                pass
                                    
            # Scale frame for the video label
            screen_w = self.video_frame.winfo_width()
            screen_h = self.video_frame.winfo_height()
            
            if screen_w > 1 and screen_h > 1:
                frame_aspect = w / h
                screen_aspect = screen_w / screen_h
                
                if screen_aspect > frame_aspect:
                    new_h = screen_h
                    new_w = int(new_h * frame_aspect)
                else:
                    new_w = screen_w
                    new_h = int(new_w / frame_aspect)
                    
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                
                bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                y_offset = (screen_h - new_h) // 2
                x_offset = (screen_w - new_w) // 2
                bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
                
                img = Image.fromarray(bg)
                self.photo = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=self.photo)
            
        self.after(self.delay, self.update_frame)
        
    def __del__(self):
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()
        pygame.mixer.quit()

if __name__ == '__main__':
    app = GestureApp("Hand Gesture App V3")
