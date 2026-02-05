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
