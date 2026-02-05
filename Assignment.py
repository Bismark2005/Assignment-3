import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


# ===================== Image Processor =====================
class ImageProcessor:
    def __init__(self):
        self.image = None
        self.original = None
        self.history = []  # undo stack (full image snapshots)
        self.redo_stack = []  # redo stack (full image snapshots)
        self.max_history = 20
        # Non-destructive adjustment params (applied on a copy when previewing)
        self.adjust_brightness = 0
        self.adjust_contrast = 1.0
        self.adjust_saturation = 1.0

    def load_image(self, path):
        # Load via OpenCV, fall back to Pillow for wider format support.
        img = cv2.imread(path)
        if img is None:
            try:
                pil = Image.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                img = None
        if img is None:
            return None
        self.image = img
        self.original = img.copy()
        self.history = []
        self.redo_stack = []
        self.adjust_brightness = 0
        self.adjust_contrast = 1.0
        self.adjust_saturation = 1.0
        return self.image

    def push_history(self):
        # Store snapshot before a destructive operation.
        if self.image is None:
            return
        self.history.append(self.image.copy())
        self.redo_stack = []
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def push_history_image(self, img):
        # Store a specific snapshot (used when committing adjustment previews).
        if img is None:
            return
        self.history.append(img.copy())
        self.redo_stack = []
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self):
        # Undo swaps current image into redo stack.
        if self.history:
            if self.image is not None:
                self.redo_stack.append(self.image.copy())
            self.image = self.history.pop()

    def redo(self):
        # Redo swaps current image into undo stack.
        if self.redo_stack:
            if self.image is not None:
                self.history.append(self.image.copy())
            self.image = self.redo_stack.pop()

    def reset(self):
        if self.original is not None:
            self.image = self.original.copy()
            self.history = []
            self.redo_stack = []

    def apply_adjustments(self):
        # Apply brightness/contrast/saturation to the current preview image.
        if self.image is None:
            return
        img = cv2.convertScaleAbs(
            self.image, alpha=self.adjust_contrast, beta=self.adjust_brightness
        )
        if self.adjust_saturation != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= self.adjust_saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        self.image = img

    def grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def blur(self, intensity):
        # Gaussian blur with adjustable intensity (odd kernel size).
        k = max(1, int(intensity))
        k = k * 2 + 1
        self.image = cv2.GaussianBlur(self.image, (k, k), 0)

    def edge_detection(self):
        # Simple edge detection for stylized outlines.
        edges = cv2.Canny(self.image, 100, 200)
        self.image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def sharpen(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.image = cv2.filter2D(self.image, -1, kernel)

    def sepia(self):
        img = self.image.astype(np.float32)
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        img = cv2.transform(img, sepia_filter)
        self.image = np.clip(img, 0, 255).astype(np.uint8)

    def rotate(self, angle):
        # Rotate by fixed angles to avoid quality loss.
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

    def crop(self, x, y, w, h):
        # Clamp the crop rectangle to the image bounds.
        h_img, w_img, _ = self.image.shape
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        self.image = self.image[y:y + h, x:x + w]

    def resize(self, scale):
        # Resize/scale the image by a factor.
        h, w, _ = self.image.shape
        self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)))


class StatusBar:
    # Small helper class to keep status updates isolated and reusable.
    def __init__(self, parent):
        self.label = tk.Label(parent, text="No image loaded", bg="#f5f2ff",
                              fg="#5c5475", anchor=tk.W)
        self.label.pack(fill=tk.X, pady=(10, 0))

    def update(self, text):
        self.label.config(text=text)


# ===================== GUI Application =====================
class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pro Image Editor")
        self.root.configure(bg="#f5f2ff")
        self.root.minsize(1000, 650)

        self.processor = ImageProcessor()
        self.file_path = None
        self.display_zoom = 1.0  # view-only zoom (does not change image pixels)
        self.slider_base = None  # base image for non-destructive adjustment preview
        self.last_slider = None  # which slider is currently driving preview
        self.drag_start = None
        self.drag_rect = None
        self.canvas_image_id = None
        # Crop tool state (non-destructive until Apply)
        self.crop_active = False
        self.crop_base_image = None
        self.crop_preview_image = None
        self.crop_angle = 0
        self.crop_rect = None  # (x0, y0, x1, y1) in image coords
        self.crop_drag_mode = None
        self.crop_overlay_items = []
        self.bg_photo = None

        self.setup_menu()
        self.setup_background()
        self.setup_ui()
        self.setup_shortcuts()

    # ---------- Layout / Style ----------
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_command(label="Save As", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Redo", command=self.redo)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        self.root.config(menu=menubar)

    def setup_shortcuts(self):
        # Core editor shortcuts.
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-S>", lambda e: self.save_as())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-r>", lambda e: self.reset())

    def setup_background(self):
        # Soft gradient background to make the UI feel less flat.
        width, height = 1000, 650
        bg = Image.new("RGB", (width, height), (245, 242, 255))
        px = bg.load()
        for y in range(height):
            for x in range(width):
                r = int(246 - (y / height) * 18 + (x / width) * 10)
                g = int(242 - (y / height) * 10 + (x / width) * 6)
                b = int(255 - (y / height) * 25 + (x / width) * 8)
                px[x, y] = (r, g, b)
        self.bg_photo = ImageTk.PhotoImage(bg)
        bg_label = tk.Label(self.root, image=self.bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    def setup_ui(self):
        # Main layout: header + left tools + right preview.
        main = tk.Frame(self.root, bg="#f5f2ff")
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        header = tk.Frame(main, bg="#f5f2ff")
        header.pack(fill=tk.X)
        title = tk.Label(header, text="Pro Image Editor", bg="#f5f2ff",
                         fg="#201a33", font=("Segoe UI", 20, "bold"))
        title.pack(anchor=tk.W)
        subtitle = tk.Label(header, text="Clean workflow, fast edits, professional feel",
                            bg="#f5f2ff", fg="#5c5475", font=("Segoe UI", 10))
        subtitle.pack(anchor=tk.W, pady=(2, 8))

        quick = tk.Frame(header, bg="#f5f2ff")
        quick.pack(fill=tk.X, pady=(0, 10))
        tk.Button(quick, text="Open", command=self.open_image,
                  bg="#ff6b6b", fg="#ffffff", relief=tk.FLAT, bd=0,
                  activebackground="#e85b5b", activeforeground="#ffffff",
                  font=("Segoe UI", 9, "bold"), padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(quick, text="Save", command=self.save_image,
                  bg="#5ad6ff", fg="#0b1d26", relief=tk.FLAT, bd=0,
                  activebackground="#42c6f2", activeforeground="#0b1d26",
                  font=("Segoe UI", 9, "bold"), padx=12, pady=6).pack(side=tk.LEFT, padx=6)
        tk.Button(quick, text="Save As", command=self.save_as,
                  bg="#ffffff", fg="#201a33", relief=tk.FLAT, bd=1,
                  highlightthickness=1, highlightbackground="#e7def8",
                  font=("Segoe UI", 9, "bold"), padx=12, pady=6).pack(side=tk.LEFT, padx=6)

        content = tk.Frame(main, bg="#f5f2ff")
        content.pack(fill=tk.BOTH, expand=True)

        left_container = tk.Frame(content, bg="#f5f2ff")
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_canvas = tk.Canvas(left_container, bg="#f5f2ff", highlightthickness=0, width=290)
        left_scroll = tk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left = tk.Frame(left_canvas, bg="#ffffff", highlightthickness=1, highlightbackground="#e7def8")
        left_canvas.create_window((0, 0), window=left, anchor="nw")

        right = tk.Frame(content, bg="#f5f2ff")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(right, bg="#ffffff", highlightthickness=1, highlightbackground="#e7def8")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#fff8f2", highlightthickness=0)
        self.canvas.pack(padx=12, pady=12)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        self._build_left_panel(left)
        left.update_idletasks()
        left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _on_left_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        left.bind("<Configure>", _on_left_configure)
        self.status = StatusBar(main)

    def _section(self, parent, text, row):
        lbl = tk.Label(parent, text=text, bg="#ffffff", fg="#201a33",
                       font=("Segoe UI", 10, "bold"))
        lbl.grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(12, 4))

    def _button(self, parent, text, cmd, row, col, ghost=False):
        if ghost:
            btn = tk.Button(parent, text=text, command=cmd,
                            bg="#ffffff", fg="#201a33", relief=tk.FLAT, bd=1,
                            highlightthickness=1, highlightbackground="#e7def8",
                            font=("Segoe UI", 9, "bold"))
        else:
            btn = tk.Button(parent, text=text, command=cmd,
                            bg="#ff6b6b", fg="#ffffff", relief=tk.FLAT, bd=0,
                            activebackground="#e85b5b", activeforeground="#ffffff",
                            font=("Segoe UI", 9, "bold"))
        btn.grid(row=row, column=col, padx=6, pady=6, sticky="ew")

    def _slider(self, parent, label, from_, to, resolution, command, row):
        s = tk.Scale(parent, from_=from_, to=to, resolution=resolution, orient=tk.HORIZONTAL,
                     label=label, command=command, bg="#ffffff", fg="#201a33",
                     troughcolor="#e9dcff", highlightthickness=0)
        s.grid(row=row, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
        return s

    def _build_left_panel(self, left):
        # Grouped tool sections to improve scanability.
        row = 0
        self._section(left, "Filters", row); row += 1
        self._button(left, "Grayscale", self.apply_grayscale, row, 0)
        self._button(left, "Blur", self.apply_blur, row, 1); row += 1
        self._button(left, "Edges", self.apply_edges, row, 0)
        self._button(left, "Sharpen", self.apply_sharpen, row, 1); row += 1
        self._button(left, "Sepia", self.apply_sepia, row, 0); row += 1
        self.blur_intensity = self._slider(left, "Blur Intensity", 1, 10, 1, self.noop, row); row += 1
        self.blur_intensity.set(3)

        self._section(left, "Transform", row); row += 1
        self._button(left, "Rotate 90", lambda: self.rotate(90), row, 0)
        self._button(left, "Rotate 180", lambda: self.rotate(180), row, 1); row += 1
        self._button(left, "Rotate 270", lambda: self.rotate(270), row, 0)
        self._button(left, "Flip H", lambda: self.flip("h"), row, 1); row += 1
        self._button(left, "Flip V", lambda: self.flip("v"), row, 0)
        self._button(left, "Crop Tool", self.toggle_crop_mode, row, 1); row += 1
        self._button(left, "Apply Crop", self.apply_crop, row, 0, ghost=True)
        self._button(left, "Cancel Crop", self.cancel_crop, row, 1, ghost=True); row += 1
        self._button(left, "Undo", self.undo, row, 0, ghost=True)
        self._button(left, "Redo", self.redo, row, 1, ghost=True); row += 1
        self._button(left, "Reset", self.reset, row, 0, ghost=True); row += 1

        self._section(left, "Adjust", row); row += 1
        self.brightness = self._slider(left, "Brightness", -100, 100, 1, self.adjust_brightness, row); row += 1
        self.contrast = self._slider(left, "Contrast", 10, 300, 1, self.adjust_contrast, row); row += 1
        self.saturation = self._slider(left, "Saturation", 0, 200, 1, self.adjust_saturation, row); row += 1

        self._button(left, "Apply Adjustments", self.apply_adjustments, row, 0, ghost=True)
        self._button(left, "Reset Adjustments", self.reset_adjustments, row, 1, ghost=True); row += 1

        self._section(left, "Crop Adjust", row); row += 1
        self.straighten = self._slider(left, "Straighten", -15, 15, 1, self.adjust_straighten, row); row += 1
        self.straighten.set(0)
        self.straighten.config(state=tk.DISABLED)

        self._section(left, "View", row); row += 1
        self.zoom = self._slider(left, "Zoom", 25, 200, 1, self.adjust_zoom, row); row += 1
        self.zoom.set(100)
        self.scale = self._slider(left, "Scale (%)", 10, 200, 1, self.noop, row); row += 1
        self.scale.set(100)
        self._button(left, "Apply Resize", self.apply_resize, row, 0, ghost=True); row += 1

        left.grid_columnconfigure(0, weight=1)
        left.grid_columnconfigure(1, weight=1)

    # ---------- Actions ----------
    def open_image(self):
        # Open an image with a wide file filter.
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.gif"), ("All Files", "*.*")]
        )
        if not path:
            return
        if self.processor.load_image(path) is None:
            messagebox.showerror("Open failed", "This file type isn't supported or can't be read.")
            return
        self.file_path = path
        self.slider_base = None
        self.last_slider = None
        self.brightness.set(0)
        self.contrast.set(100)
        self.saturation.set(100)
        self.zoom.set(100)
        self.blur_intensity.set(3)
        self.scale.set(100)
        self.update_image()

    def save_image(self):
        # Save to existing file or prompt Save As.
        if self.processor.image is None:
            return
        if not self.file_path:
            self.save_as()
            return
        self._commit_adjustments_before_action()
        cv2.imwrite(self.file_path, self.processor.image)
        messagebox.showinfo("Saved", "Image saved successfully")

    def save_as(self):
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp")])
        if path:
            self._commit_adjustments_before_action()
            cv2.imwrite(path, self.processor.image)
            self.file_path = path

    def apply_grayscale(self):
        # Filters are destructive, so we commit any preview adjustments first.
        self._commit_adjustments_before_action()
        self.processor.push_history()
        self.processor.grayscale()
        self.slider_base = None
        self.update_image()

    def apply_blur(self):
        self._commit_adjustments_before_action()
        self.processor.push_history()
        self.processor.blur(self.blur_intensity.get())
        self.slider_base = None
        self.update_image()

    def apply_edges(self):
        self._commit_adjustments_before_action()
        self.processor.push_history()
        self.processor.edge_detection()
        self.slider_base = None
        self.update_image()

    def apply_sharpen(self):
        self._commit_adjustments_before_action()
        self.processor.push_history()
        self.processor.sharpen()
        self.slider_base = None
        self.update_image()

    def apply_sepia(self):
        self._commit_adjustments_before_action()
        self.processor.push_history()
        self.processor.sepia()
        self.slider_base = None
        self.update_image()

    def rotate(self, angle):
        self._commit_adjustments_before_action()
        self.processor.push_history()
        self.processor.rotate(angle)
        self.slider_base = None
        self.update_image()

    def flip(self, mode):
        self._commit_adjustments_before_action()
        self.processor.push_history()
        self.processor.flip(mode)
        self.slider_base = None
        self.update_image()

    def undo(self):
        self.processor.undo()
        self.slider_base = None
        self.update_image()

    def redo(self):
        self.processor.redo()
        self.slider_base = None
        self.update_image()

    def reset(self):
        if not messagebox.askyesno("Reset", "Reset the image to its original state?"):
            return
        self.processor.reset()
        self.slider_base = None
        self.brightness.set(0)
        self.contrast.set(100)
        self.saturation.set(100)
        self.zoom.set(100)
        self.blur_intensity.set(3)
        self.scale.set(100)
        self.update_image()

    def open_crop_dialog(self):
        if self.processor.image is None:
            return
        self._commit_adjustments_before_action()
        dialog = tk.Toplevel(self.root)
        dialog.title("Crop")
        dialog.configure(bg="#ffffff")

        tk.Label(dialog, text="X", bg="#ffffff").grid(row=0, column=0)
        tk.Label(dialog, text="Y", bg="#ffffff").grid(row=0, column=2)
        tk.Label(dialog, text="Width", bg="#ffffff").grid(row=1, column=0)
        tk.Label(dialog, text="Height", bg="#ffffff").grid(row=1, column=2)

        x_entry = tk.Entry(dialog, width=6)
        y_entry = tk.Entry(dialog, width=6)
        w_entry = tk.Entry(dialog, width=6)
        h_entry = tk.Entry(dialog, width=6)

        x_entry.insert(0, "0")
        y_entry.insert(0, "0")
        h_img, w_img, _ = self.processor.image.shape
        w_entry.insert(0, str(w_img))
        h_entry.insert(0, str(h_img))

        x_entry.grid(row=0, column=1)
        y_entry.grid(row=0, column=3)
        w_entry.grid(row=1, column=1)
        h_entry.grid(row=1, column=3)

        def apply_crop():
            try:
                x = int(x_entry.get())
                y = int(y_entry.get())
                w = int(w_entry.get())
                h = int(h_entry.get())
            except ValueError:
                messagebox.showerror("Invalid input", "Please enter integer values.")
                return
            self.processor.push_history()
            self.processor.crop(x, y, w, h)
            self.update_image()
            dialog.destroy()

        tk.Button(dialog, text="Apply", command=apply_crop, bg="#ff6b6b", fg="#ffffff",
                  relief=tk.FLAT, bd=0).grid(row=2, column=0, columnspan=4, pady=8)

    def adjust_brightness(self, value):
        self._update_adjustments("brightness")

    def adjust_contrast(self, value):
        self._update_adjustments("contrast")

    def adjust_saturation(self, value):
        self._update_adjustments("saturation")

    def adjust_zoom(self, value):
        # Zoom only affects the display.
        self.display_zoom = float(value) / 100.0
        self.update_image()

    def apply_resize(self):
        # Resize using the scale slider (destructive).
        if self.processor.image is None:
            return
        self._commit_adjustments_before_action()
        self.processor.push_history()
        scale = float(self.scale.get()) / 100.0
        self.processor.resize(scale)
        self.scale.set(100)
        self.update_image()

    def noop(self, _value):
        # Placeholder callback for sliders that apply on button press.
        pass

    def toggle_crop_mode(self):
        # Enter crop mode with handles + straighten preview.
        if self.processor.image is None:
            return
        if self.crop_active:
            self.cancel_crop()
            return
        self._commit_adjustments_before_action()
        self.crop_active = True
        self.crop_base_image = self.processor.image.copy()
        self.crop_angle = 0
        self.straighten.config(state=tk.NORMAL)
        self.straighten.set(0)
        self._update_crop_preview()
        self._init_crop_rect()
        self.status.update("Crop mode: drag to resize, drag inside to move, adjust straighten slider")
        self.update_image()

    def _update_crop_preview(self):
        # Rotate the base image for straighten preview (non-destructive).
        if self.crop_base_image is None:
            return
        angle = self.crop_angle
        img = self.crop_base_image
        if angle == 0:
            self.crop_preview_image = img.copy()
            self.processor.image = self.crop_preview_image
            return
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(m[0, 0])
        sin = abs(m[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        m[0, 2] += (new_w / 2) - center[0]
        m[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(img, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self.crop_preview_image = rotated
        self.processor.image = self.crop_preview_image

    def _init_crop_rect(self):
        if self.processor.image is None:
            return
        h, w = self.processor.image.shape[:2]
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        self.crop_rect = (margin_x, margin_y, w - margin_x, h - margin_y)

    def adjust_straighten(self, value):
        # Straighten rotates the preview and resets crop rect.
        if not self.crop_active:
            return
        self.crop_angle = int(value)
        self._update_crop_preview()
        self._init_crop_rect()
        self.update_image()

    def apply_crop(self):
        # Commit the crop to the image and exit crop mode.
        if not self.crop_active or self.processor.image is None or self.crop_rect is None:
            return
        x0, y0, x1, y1 = self.crop_rect
        self.processor.push_history()
        self.processor.crop(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        self.crop_active = False
        self.crop_base_image = None
        self.crop_preview_image = None
        self.crop_rect = None
        self.straighten.set(0)
        self.straighten.config(state=tk.DISABLED)
        self.update_image()

    def cancel_crop(self):
        if not self.crop_active:
            return
        if self.crop_base_image is not None:
            self.processor.image = self.crop_base_image.copy()
        self.crop_active = False
        self.crop_base_image = None
        self.crop_preview_image = None
        self.crop_rect = None
        self.straighten.set(0)
        self.straighten.config(state=tk.DISABLED)
        self.update_image()

    def _update_adjustments(self, source):
        # Live preview for adjustments; commits only on Apply.
        if self.processor.image is None:
            return
        if self.last_slider != source:
            self.slider_base = None
        if self.slider_base is None:
            self.slider_base = self.processor.image.copy()
        self.processor.image = self.slider_base.copy()
        self.processor.adjust_brightness = int(self.brightness.get())
        self.processor.adjust_contrast = float(self.contrast.get()) / 100.0
        self.processor.adjust_saturation = float(self.saturation.get()) / 100.0
        self.processor.apply_adjustments()
        self.last_slider = source
        self.update_image()

    def _commit_adjustments_before_action(self):
        # Bake current preview into the history before destructive ops.
        if self.slider_base is None:
            return
        self.processor.push_history_image(self.slider_base)
        self.slider_base = None
        self.last_slider = None
        self.brightness.set(0)
        self.contrast.set(100)
        self.saturation.set(100)

    def apply_adjustments(self):
        if self.slider_base is None:
            return
        self.processor.push_history_image(self.slider_base)
        self.slider_base = None
        self.last_slider = None
        self.brightness.set(0)
        self.contrast.set(100)
        self.saturation.set(100)
        self.update_image()

    def reset_adjustments(self):
        if self.slider_base is None:
            return
        self.processor.image = self.slider_base.copy()
        self.slider_base = None
        self.last_slider = None
        self.brightness.set(0)
        self.contrast.set(100)
        self.saturation.set(100)
        self.update_image()

    def update_image(self):
        # Render the current image to the canvas and update status text.
        if self.processor.image is None:
            self.status.update("No image loaded")
            self.canvas.delete("all")
            return
        img = cv2.cvtColor(self.processor.image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.display_zoom != 1.0:
            w, h = img.size
            img = img.resize((int(w * self.display_zoom), int(h * self.display_zoom)))
        self.canvas.config(width=img.size[0], height=img.size[1])
        img = ImageTk.PhotoImage(img)
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=img)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=img)
        self.canvas.image = img
        self._draw_crop_overlay()
        h, w = self.processor.image.shape[:2]
        name = os.path.basename(self.file_path) if self.file_path else "Untitled"
        self.status.update(f"{name} | {w} x {h} | Zoom {int(self.display_zoom*100)}%")

    def _draw_crop_overlay(self):
        # Draw crop rectangle and resize handles.
        for item in self.crop_overlay_items:
            self.canvas.delete(item)
        self.crop_overlay_items = []
        if not self.crop_active or self.crop_rect is None:
            return
        x0, y0, x1, y1 = self.crop_rect
        x0d = x0 * self.display_zoom
        y0d = y0 * self.display_zoom
        x1d = x1 * self.display_zoom
        y1d = y1 * self.display_zoom
        rect_id = self.canvas.create_rectangle(
            x0d, y0d, x1d, y1d, outline="#ff6b6b", width=2
        )
        self.crop_overlay_items.append(rect_id)
        size = 6
        for cx, cy in [(x0d, y0d), (x1d, y0d), (x0d, y1d), (x1d, y1d)]:
            handle = self.canvas.create_rectangle(
                cx - size, cy - size, cx + size, cy + size, fill="#ff6b6b", outline="#ffffff"
            )
            self.crop_overlay_items.append(handle)

    # ---------- Drag-to-crop ----------
    def on_drag_start(self, event):
        # Detect which crop handle is grabbed.
        if self.processor.image is None or not self.crop_active:
            return
        self.drag_start = (event.x, event.y)
        self.crop_drag_mode = self._hit_test_crop(event.x, event.y)
        if self.crop_drag_mode == "new" or self.crop_rect is None:
            x = event.x / self.display_zoom
            y = event.y / self.display_zoom
            self.crop_rect = (x, y, x, y)

    def on_drag_move(self, event):
        # Resize or move crop rectangle based on drag mode.
        if self.drag_start is None or not self.crop_active or self.crop_rect is None:
            return
        x0d, y0d = self.drag_start
        x0, y0, x1, y1 = self.crop_rect
        ex = event.x / self.display_zoom
        ey = event.y / self.display_zoom
        dx = (event.x - x0d) / self.display_zoom
        dy = (event.y - y0d) / self.display_zoom

        if self.crop_drag_mode == "move":
            x0 += dx
            y0 += dy
            x1 += dx
            y1 += dy
            self.drag_start = (event.x, event.y)
        elif self.crop_drag_mode == "nw":
            x0, y0 = ex, ey
        elif self.crop_drag_mode == "ne":
            x1, y0 = ex, ey
        elif self.crop_drag_mode == "sw":
            x0, y1 = ex, ey
        elif self.crop_drag_mode == "se":
            x1, y1 = ex, ey
        else:
            x1, y1 = ex, ey

        x0, y0, x1, y1 = self._normalize_rect(x0, y0, x1, y1)
        self.crop_rect = self._clamp_rect(x0, y0, x1, y1)
        self._draw_crop_overlay()

    def on_drag_end(self, event):
        if self.drag_start is None or not self.crop_active:
            return
        self.drag_start = None
        self.crop_drag_mode = None

    def _hit_test_crop(self, x, y):
        # Determine which handle (or move) the cursor is over.
        if self.crop_rect is None:
            return "new"
        x0, y0, x1, y1 = self.crop_rect
        x0d, y0d = x0 * self.display_zoom, y0 * self.display_zoom
        x1d, y1d = x1 * self.display_zoom, y1 * self.display_zoom
        handle = 8
        if abs(x - x0d) <= handle and abs(y - y0d) <= handle:
            return "nw"
        if abs(x - x1d) <= handle and abs(y - y0d) <= handle:
            return "ne"
        if abs(x - x0d) <= handle and abs(y - y1d) <= handle:
            return "sw"
        if abs(x - x1d) <= handle and abs(y - y1d) <= handle:
            return "se"
        if x0d < x < x1d and y0d < y < y1d:
            return "move"
        return "new"

    def _normalize_rect(self, x0, y0, x1, y1):
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def _clamp_rect(self, x0, y0, x1, y1):
        if self.processor.image is None:
            return x0, y0, x1, y1
        h, w = self.processor.image.shape[:2]
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))
        return x0, y0, x1, y1


# ===================== Main =====================
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
