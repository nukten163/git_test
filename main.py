import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import time
import numpy as np
import collections

class GestureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Make the window fullscreen
        self.window.attributes("-fullscreen", True)
        self.window.bind("<Escape>", lambda e: self.window.attributes("-fullscreen", False))
        self.window.bind("<q>", lambda e: self.window.destroy())
        
        # Load the emoji image
        self.emoji_path = "assets/emoji.png"
        try:
            self.emoji_img = cv2.imread(self.emoji_path, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print("Could not load emoji:", e)
            self.emoji_img = None
            
        # State variables
        self.is_active = False
        self.gesture_detected = False
        self.gesture_detect_time = 0
        self.emoji_duration = 3.0

        # Hand tracking history
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
            base_options=BaseOptions(model_asset_path='assets/hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2
        )
        self.hands = HandLandmarker.create_from_options(hand_options)

        # Face Detector
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        face_options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path='assets/face_detector.task'),
            running_mode=VisionRunningMode.IMAGE
        )
        self.face_detection = FaceDetector.create_from_options(face_options)
        
        # Open video source
        self.vid = cv2.VideoCapture(0)
        width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Frame container
        self.canvas = tk.Canvas(window, width=self.window.winfo_screenwidth(), height=self.window.winfo_screenheight(), bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.btn_activate = tk.Button(self.window, text="Activate", command=self.toggle_activate, 
                                      font=("Arial", 20, "bold"), bg="#4CAF50", fg="white", 
                                      activebackground="#45a049", padx=20, pady=10)
        self.btn_activate.place(x=20, y=20)
        
        self.btn_quit = tk.Button(self.window, text="Quit", command=self.window.destroy, 
                                      font=("Arial", 14), bg="#f44336", fg="white", 
                                      activebackground="#d32f2f", padx=10, pady=5)
        self.btn_quit.place(x=20, y=90)
        
        self.delay = 15
        self.update_frame()
        self.window.mainloop()
        
    def toggle_activate(self):
        self.is_active = not self.is_active
        if self.is_active:
            self.btn_activate.config(text="Deactivate", bg="#f44336")
        else:
            self.btn_activate.config(text="Activate", bg="#4CAF50")
            self.gesture_detected = False
            
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
        
        threshold = 0.05
        if (left_dy > threshold and right_dy < -threshold) or (left_dy < -threshold and right_dy > threshold):
            return True
        return False

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
                    # Draw boxes for all detected hands for troubleshooting
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
                            
                if left_y is not None and right_y is not None:
                    self.left_y_history.append(left_y)
                    self.right_y_history.append(right_y)
                    if self.check_6_7_gesture():
                        self.gesture_detected = True
                        self.gesture_detect_time = time.time()
                        self.left_y_history.clear()
                        self.right_y_history.clear()
                else:
                    self.left_y_history.clear()
                    self.right_y_history.clear()
                    
                # Face Detection & Emoji
                if self.gesture_detected:
                    if time.time() - self.gesture_detect_time > self.emoji_duration:
                        self.gesture_detected = False
                    else:
                        face_result = self.face_detection.detect(mp_image)
                        if face_result.detections and self.emoji_img is not None:
                            detection = face_result.detections[0]
                            bbox = detection.bounding_box
                            
                            # Original face box
                            orig_x, orig_y = bbox.origin_x, bbox.origin_y
                            orig_w, orig_h = bbox.width, bbox.height
                            
                            # Draw face box for troubleshooting
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
                                    resized_emoji = cv2.resize(self.emoji_img, (bw, bh))
                                    emoji_rgba = cv2.cvtColor(resized_emoji, cv2.COLOR_BGRA2RGBA)
                                    self.overlay_image_alpha(frame_rgb, emoji_rgba, bx, by)
                                    
                                    # Add "six-seven" caption above emoji
                                    text = "six-seven"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 1
                                    thickness = 2
                                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                                    text_x = bx + (bw - text_size[0]) // 2
                                    text_y = max(0, by - 10)
                                    
                                    # Outline for better visibility
                                    cv2.putText(frame_rgb, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                                    cv2.putText(frame_rgb, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                                except Exception as e:
                                    pass
                                    
            screen_w = self.window.winfo_screenwidth()
            screen_h = self.window.winfo_screenheight()
            
            frame_aspect = w / h
            screen_aspect = screen_w / screen_h
            
            if screen_aspect > frame_aspect:
                new_h = screen_h
                new_w = int(new_h * frame_aspect)
            else:
                new_w = screen_w
                new_h = int(new_w / frame_aspect)
                
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            img = Image.fromarray(frame_resized)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(screen_w//2, screen_h//2, image=self.photo, anchor=tk.CENTER)
            
        self.window.after(self.delay, self.update_frame)
        
    def __del__(self):
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()

if __name__ == '__main__':
    root = tk.Tk()
    app = GestureApp(root, "Hand Gesture App")
