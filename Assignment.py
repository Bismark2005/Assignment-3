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
