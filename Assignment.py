python
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

if __name__ == "__main__":
    root = tk.Tk()
    root.mainloop()
    
class ImageProcessor:
    def __init__(self):
        self.image = None
        self.original = None
        self.history = []
        self.redo_stack = []
        self.max_history = 20

        self.adjust_brightness = 0
        self.adjust_contrast = 1.0
        self.adjust_saturation = 1.0
def load_image(self, path):
    img = cv2.imread(path)
    if img is None:
        try:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    self.image = img
    self.original = img.copy()
    self.history.clear()
    self.redo_stack.clear()
    return img
def push_history(self):
    if self.image is None:
        return
    self.history.append(self.image.copy())
    self.redo_stack.clear()
    if len(self.history) > self.max_history:
        self.history.pop(0)

def undo(self):
    if self.history:
        self.redo_stack.append(self.image.copy())
        self.image = self.history.pop()

def redo(self):
    if self.redo_stack:
        self.history.append(self.image.copy())
        self.image = self.redo_stack.pop()
def grayscale(self):
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

def blur(self):
    self.image = cv2.GaussianBlur(self.image, (7, 7), 0)

def edge_detection(self):
    edges = cv2.Canny(self.image, 100, 200)
    self.image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def sharpen(self):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    self.image = cv2.filter2D(self.image, -1, kernel)
def rotate(self, angle):
    if angle == 90:
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        self.image = cv2.rotate(self.image, cv2.ROTATE_180)
    elif angle == 270:
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def flip(self, mode):
    if mode == "h":
        self.image = cv2.flip(self.image, 1)
    elif mode == "v":
        self.image = cv2.flip(self.image, 0)
def apply_adjustments(self):
    img = cv2.convertScaleAbs(
        self.image,
        alpha=self.adjust_contrast,
        beta=self.adjust_brightness
    )

    if self.adjust_saturation != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= self.adjust_saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    self.image = img
class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pro Image Editor")
        self.root.minsize(1000, 650)

        self.processor = ImageProcessor()
        self.file_path = None
