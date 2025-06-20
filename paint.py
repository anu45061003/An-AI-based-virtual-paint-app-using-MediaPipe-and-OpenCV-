import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize canvas dimensions
WIDTH, HEIGHT = 640, 480
brush_thickness = 5
eraser_thickness = 50

# Color Palette
color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 255, 255)]
color_index = 0
draw_color = color_palette[color_index]

class PaintApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Gesture Paint App")
        self.cap = cv2.VideoCapture(0)
        self.canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        self.label = tk.Label(self.window)
        self.label.pack()

        self.previous_point = None
        self.drawing = False
        self.eraser_mode = False
        self.prev_x = None
        self.last_color_change = time.time()
        self.show_color_time = 0

        self.running = True
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        self.thread = threading.Thread(target=self.update)
        self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        self.window.destroy()

    def update(self):
        global draw_color, color_index
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[8]
                    x, y = int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT)

                    finger_states = self.get_finger_states(hand_landmarks)

                    # === Drawing Mode (Index finger only) ===
                    if finger_states[1] and not any(finger_states[2:]):
                        self.drawing = True
                        if self.previous_point:
                            if self.eraser_mode:
                                cv2.line(self.canvas, self.previous_point, (x, y), (0, 0, 0), eraser_thickness)
                            else:
                                cv2.line(self.canvas, self.previous_point, (x, y), draw_color, brush_thickness)
                        self.previous_point = (x, y)
                    else:
                        self.previous_point = None
                        self.drawing = False

                    # === Color Change Mode (Index + Middle only) ===
                    if finger_states[1] and finger_states[2] and not any(finger_states[3:]):
                        if self.prev_x is not None:
                            dx = x - self.prev_x
                            if abs(dx) > 40 and time.time() - self.last_color_change > 1:
                                if dx > 0:
                                    color_index = (color_index + 1) % len(color_palette)
                                else:
                                    color_index = (color_index - 1) % len(color_palette)
                                draw_color = color_palette[color_index]
                                self.last_color_change = time.time()
                                self.show_color_time = time.time()
                        self.prev_x = x
                    else:
                        self.prev_x = None

                    # === Toggle eraser with all fingers down ===
                    if not any(finger_states):
                        self.eraser_mode = not self.eraser_mode

            combined = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)

            # Show current color as a circle (when recently changed)
            if time.time() - self.show_color_time < 1:
                cv2.circle(combined, (30, 30), 20, draw_color, -1)

            img = Image.fromarray(combined)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

    def get_finger_states(self, hand_landmarks):
        fingers = []
        # Thumb
        fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
        # Index
        fingers.append(hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y)
        # Middle
        fingers.append(hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y)
        # Ring
        fingers.append(hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y)
        # Pinky
        fingers.append(hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y)
        return fingers

# Start GUI
root = tk.Tk()
app = PaintApp(root)
root.mainloop()
