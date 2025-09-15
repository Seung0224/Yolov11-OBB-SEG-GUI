import os
import re
import math
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO

DEFAULT_MODEL_PATH = r"D:\DLP\OS\Model\SEG\SEG.onnx"  # or .pt
CONF_DEFAULT = 0.25
IOU_DEFAULT = 0.45
LINE_THICKNESS = 2
FONT_SCALE = 0.6
PADDING = 6
MASK_THR_DEFAULT = 0.5
ALPHA_DEFAULT = 0.45
WINDOW_TITLE = "YOLOv11"


# CPU Version
# pip install --upgrade ultralytics pillow opencv-python numpy torch

# GPU Version
# pip uninstall -y torch torchvision torchaudio
# pip cache purge

# Onnx Model Export
# yolo export model="D:\DLP\OS\Model\SEG\SEG.pt" task=segment format=onnx device=cpu imgsz=640 dynamic=False

# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# =========================
# 이미지 패널 (Canvas + Scrollbar + Zoom/Pan)
# =========================
class ImagePane(tk.Frame):
    def __init__(self, master, title=""):
        super().__init__(master)

        # 제목 바
        titlebar = tk.Frame(self)
        titlebar.pack(fill=tk.X, padx=4, pady=(4, 0))
        self.title_label = tk.Label(titlebar, text=title, anchor="w")
        self.title_label.pack(side=tk.LEFT)

        # 캔버스 + 스크롤
        body = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = tk.Canvas(body, bg="black", highlightthickness=0)
        self.hbar = tk.Scrollbar(body, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = tk.Scrollbar(body, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")

        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        # 상태
        self.img_bgr = None
        self._pil = None
        self._tk = None
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self._image_id = None
        self._drag_origin = None

        # 바인딩
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Configure>", lambda e: self.render())  # 리사이즈 시 재렌더
        # Ctrl + 휠 확대/축소 (Windows/Unix 공통 처리)
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel)       # Windows
        self.canvas.bind("<Control-Button-4>", self._on_wheel_linux)   # Linux up
        self.canvas.bind("<Control-Button-5>", self._on_wheel_linux)   # Linux down

        self.on_zoom_changed = None  # 외부 콜백(좌/우 동기화용)

    def set_title(self, text: str):
        self.title_label.config(text=text)

    def set_image(self, img_bgr: np.ndarray | None, placeholder: str | None = None):
        """BGR np.ndarray 설정. None이면 placeholder 텍스트 표시."""
        self.img_bgr = img_bgr
        if img_bgr is None:
            self._pil = None
            self._tk = None
            self._draw_placeholder(placeholder or "No image")
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self._pil = Image.fromarray(img_rgb)
            self._tk = None
            self.render()

    def _draw_placeholder(self, text="No image"):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 400
        h = self.canvas.winfo_height() or 300
        self.canvas.config(scrollregion=(0, 0, w, h))
        self._image_id = None
        self.canvas.create_text(
            w // 2, h // 2,
            text=text,
            fill="#CCCCCC",
            font=("Segoe UI", 14, "bold")
        )

    def render(self):
        if self._pil is None:
            self._draw_placeholder()
            return

        base_w, base_h = self._pil.size
        target_w = max(1, int(base_w * self.zoom))
        target_h = max(1, int(base_h * self.zoom))

        disp = self._pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
        self._tk = ImageTk.PhotoImage(disp)

        self.canvas.delete("all")
        self._image_id = self.canvas.create_image(0, 0, image=self._tk, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, target_w, target_h))

    def zoom_in(self, step=0.1):
        self.set_zoom(self.zoom * (1.0 + step))

    def zoom_out(self, step=0.1):
        self.set_zoom(self.zoom / (1.0 + step))

    def reset_zoom(self):
        self.zoom = 1.0
        self.render()

    def fit_to_window(self, margin=8):
        """현재 캔버스 크기에 맞춰 '가득' 보이도록 줌 자동 계산."""
        if self._pil is None:
            return
        cw = max(1, self.canvas.winfo_width() - margin)
        ch = max(1, self.canvas.winfo_height() - margin)
        iw, ih = self._pil.size
        if iw == 0 or ih == 0:
            return
        z = min(cw / iw, ch / ih)
        z = max(self.min_zoom, min(self.max_zoom, z))
        self.zoom = z
        self.render()

    def set_zoom(self, value: float):
        """줌 설정 + 변경 이벤트 콜백(동기화용)"""
        value = max(self.min_zoom, min(self.max_zoom, float(value)))
        if abs(value - self.zoom) > 1e-6:
            self.zoom = value
            self.render()
            if callable(self.on_zoom_changed):
                self.on_zoom_changed(self.zoom, origin=self)

    def _on_press(self, event):
        # 패닝 시작
        self._drag_origin = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def _on_drag(self, event):
        # 드래그로 패닝
        if self._drag_origin is not None:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_wheel(self, event):
        # Windows: event.delta 양수=up, 음수=down (Control과 함께 바인딩됨)
        if event.delta > 0:
            self.zoom_in(0.2)
        else:
            self.zoom_out(0.2)

    def _on_wheel_linux(self, event):
        # Linux: Button-4(up), Button-5(down)
        if event.num == 4:
            self.zoom_in(0.2)
        elif event.num == 5:
            self.zoom_out(0.2)


# =========================
# Overlay SEG
# =========================
def draw_segmentation_on_image(
    img_bgr,
    result,
    names,
    thr=MASK_THR_DEFAULT,
    alpha=ALPHA_DEFAULT,
    line_thickness=LINE_THICKNESS,
    font_scale=FONT_SCALE,
    padding=PADDING,
):
    base = img_bgr.copy()
    h, w = base.shape[:2]

    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        cv2.putText(base, "No segmentation masks", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return base

    mask_data = masks.data
    mask_np = mask_data.cpu().numpy() if hasattr(mask_data, "cpu") else np.asarray(mask_data)

    boxes = getattr(result, "boxes", None)
    cls_ids = confs = None
    if boxes is not None:
        if getattr(boxes, "cls", None) is not None:
            cls_ids = boxes.cls.int().cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.astype(int)
        if getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "cpu") else boxes.conf

    N = mask_np.shape[0]
    cx = np.zeros(N, dtype=float)
    cy = np.zeros(N, dtype=float)
    for i in range(N):
        m = (mask_np[i] > thr).astype(np.uint8)
        ys, xs = np.where(m == 1)
        if len(xs) > 0:
            cx[i] = xs.mean()
            cy[i] = ys.mean()
        else:
            cx[i] = i
            cy[i] = i

    order = np.lexsort((cx, cy))
    mask_np = mask_np[order]
    if isinstance(cls_ids, np.ndarray):
        cls_ids = cls_ids[order]
    if isinstance(confs, np.ndarray):
        confs = confs[order]

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(mask_np.shape[0]):
        m = (mask_np[i] > thr).astype(np.uint8)
        if m.max() == 0:
            continue

        cls_i = int(cls_ids[i]) if (isinstance(cls_ids, np.ndarray) and i < len(cls_ids)) else i
        rng = np.random.default_rng(seed=int(cls_i * 123457))
        color = rng.integers(low=64, high=255, size=3, dtype=np.uint8).tolist()

        colored = np.zeros_like(overlay)
        colored[m == 1] = color
        overlay = cv2.add(overlay, colored)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            cv2.polylines(base, [c], isClosed=True, color=tuple(int(x) for x in color), thickness=line_thickness)

            M = cv2.moments(c)
            if M["m00"] > 0:
                tx = int(M["m10"] / M["m00"])
                ty = int(M["m01"] / M["m00"])
            else:
                x, y, ww, hh = cv2.boundingRect(c)
                tx, ty = x + ww // 2, y + hh // 2

            label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else str(cls_i)
            score = float(confs[i]) if (isinstance(confs, np.ndarray) and i < len(confs)) else None
            caption = f"{label} {score:.2f}" if score is not None else f"{label}"

            (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            x1 = max(0, min(tx, w - tw - padding * 2))
            y1 = max(th + baseline + padding, min(ty, h - 2))
            cv2.rectangle(
                base,
                (x1, y1 - th - baseline - padding),
                (x1 + tw + padding * 2, y1),
                tuple(int(x) for x in color),
                -1,
            )
            cv2.putText(
                base,
                caption,
                (x1 + padding, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    base = cv2.addWeighted(base, 1.0, overlay, ALPHA_DEFAULT, 0)
    return base

# =========================
# Overlay OBB
# =========================
def draw_obb_on_image(
    img_bgr,
    result,
    names,
    line_thickness=LINE_THICKNESS,
    font_scale=FONT_SCALE,
    padding=PADDING,
):
    base = img_bgr.copy()
    h, w = base.shape[:2]

    obb = getattr(result, "obb", None)
    if obb is None:
        cv2.putText(base, "No OBB results", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return base

    polys = confs = clss = None

    if getattr(obb, "xyxyxyxy", None) is not None:
        polys = obb.xyxyxyxy
        polys = polys.cpu().numpy() if hasattr(polys, "cpu") else polys
        confs = obb.conf.cpu().numpy() if getattr(obb, "conf", None) is not None and hasattr(obb, "conf") and hasattr(obb.conf, "cpu") else getattr(obb, "conf", None)
        clss = obb.cls.cpu().numpy().astype(int) if getattr(obb, "cls", None) is not None and hasattr(obb, "cls") and hasattr(obb, "cpu") else getattr(obb, "cls", None)

    elif getattr(obb, "xywhr", None) is not None:
        xywhr = obb.xywhr
        xywhr = xywhr.cpu().numpy() if hasattr(xywhr, "cpu") else xywhr
        confs = obb.conf.cpu().numpy() if getattr(obb, "conf", None) is not None and hasattr(obb, "conf") and hasattr(obb.conf, "cpu") else getattr(obb, "conf", None)
        clss = obb.cls.cpu().numpy().astype(int) if getattr(obb, "cls", None) is not None and hasattr(obb, "cls") and hasattr(obb, "cpu") else getattr(obb, "cls", None)

        polys_list = []
        for row in xywhr:
            xc, yc, ww, hh, theta = row[:5]
            deg = float(theta)
            if abs(deg) <= math.pi * 2:
                deg = deg * 180.0 / math.pi
            rect = ((float(xc), float(yc)), (float(ww), float(hh)), float(deg))
            pts = cv2.boxPoints(rect)
            polys_list.append(pts.reshape(-1))
        polys = np.array(polys_list, dtype=np.float32)

    if polys is None or len(polys) == 0:
        cv2.putText(base, "No OBB results", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return base

    for i in range(len(polys)):
        poly = polys[i]
        pts = poly.reshape(-1, 1, 2).astype(int)

        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, w - 1)
        pts[:, 0, 1] = np.clip(pts[:, 0, 1], 0, h - 1)

        c = int(clss[i]) if isinstance(clss, np.ndarray) and i < len(clss) else -1
        conf = float(confs[i]) if isinstance(confs, np.ndarray) and i < len(confs) else None

        seed_val = int((c if c >= 0 else i) * 123457)
        rng = np.random.default_rng(seed=seed_val)
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))

        cv2.polylines(base, [pts], isClosed=True, color=color, thickness=line_thickness)

        label = names.get(c, str(c)) if isinstance(names, dict) else str(c)
        caption = f"{label} {conf:.2f}" if conf is not None else f"{label}"

        x1, y1 = int(pts[0][0][0]), int(pts[0][0][1])
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LINE_THICKNESS)
        cv2.rectangle(base, (x1, y1 - th - baseline - padding), (x1 + tw + padding * 2, y1), color, -1)
        cv2.putText(base, caption, (x1 + padding, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255),
                    LINE_THICKNESS, cv2.LINE_AA)

    return base

# =========================
# 메인 앱
# =========================
class YoloApp:
    def __init__(self, master):
        self.mode = None
        self.master = master
        master.title(WINDOW_TITLE)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = DEFAULT_MODEL_PATH
        self.model = None

        # engine flags
        self.is_engine = str(self.model_path).lower().endswith(".engine")
        self.engine_imgsz = None

        # state
        self.image_path = None
        self.src_bgr = None
        self.vis_bgr = None
        self.names = {}

        # ========== 상단 UI ==========
        top = tk.Frame(master)
        top.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.model_label = tk.Label(top, text=f"Model: {self.model_path}  (Device: {self.device})", anchor="w")
        self.model_label.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

        self.btn_select_model = tk.Button(top, text="Change Model...", command=self.change_model)
        self.btn_select_model.pack(side=tk.RIGHT)

        actions = tk.Frame(master)
        actions.pack(fill=tk.X, padx=8, pady=(4, 2))

        self.btn_load = tk.Button(actions, text="Load Image", width=14, command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        self.btn_infer = tk.Button(actions, text="Run Inference", width=18, command=self.run_inference, state=tk.DISABLED)
        self.btn_infer.pack(side=tk.LEFT, padx=4)

        self.btn_save = tk.Button(actions, text="Save Result", width=14, command=self.save_result, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.btn_clear = tk.Button(actions, text="Clear", width=10, command=self.clear_view, state=tk.NORMAL)
        self.btn_clear.pack(side=tk.LEFT, padx=4)

        # 파라미터
        params = tk.Frame(master)
        params.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.conf_label = tk.Label(params, text="Conf:")
        self.conf_label.pack(side=tk.LEFT, padx=(4, 2))
        self.conf_var = tk.DoubleVar(value=CONF_DEFAULT)
        self.conf_entry = tk.Entry(params, textvariable=self.conf_var, width=6)
        self.conf_entry.pack(side=tk.LEFT, padx=(0, 8))

        self.iou_label = tk.Label(params, text="IoU:")
        self.iou_label.pack(side=tk.LEFT, padx=(8, 2))
        self.iou_var = tk.DoubleVar(value=IOU_DEFAULT)
        self.iou_entry = tk.Entry(params, textvariable=self.iou_var, width=6)
        self.iou_entry.pack(side=tk.LEFT, padx=(0, 8))

        self.maskthr_label = tk.Label(params, text="MaskThr:")
        self.maskthr_label.pack(side=tk.LEFT, padx=(8, 2))
        self.maskthr_var = tk.DoubleVar(value=MASK_THR_DEFAULT)
        self.maskthr_entry = tk.Entry(params, textvariable=self.maskthr_var, width=6)
        self.maskthr_entry.pack(side=tk.LEFT, padx=(0, 8))

        # 줌 컨트롤
        zoombar = tk.Frame(master)
        zoombar.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(zoombar, text="Zoom:").pack(side=tk.LEFT, padx=(4, 6))
        self.btn_zoom_in = tk.Button(zoombar, text="Zoom In (+)", command=lambda: self._zoom_both("in"))
        self.btn_zoom_in.pack(side=tk.LEFT, padx=2)
        self.btn_zoom_out = tk.Button(zoombar, text="Zoom Out (-)", command=lambda: self._zoom_both("out"))
        self.btn_zoom_out.pack(side=tk.LEFT, padx=2)
        self.btn_fit = tk.Button(zoombar, text="Fit", command=self._fit_both)
        self.btn_fit.pack(side=tk.LEFT, padx=2)
        self.btn_reset = tk.Button(zoombar, text="Reset (1:1)", command=self._reset_both)
        self.btn_reset.pack(side=tk.LEFT, padx=2)

        # ========== 중앙 분할 뷰 ==========
        self.split = tk.PanedWindow(master, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.split.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.left_pane = ImagePane(self.split, title="Original")
        self.right_pane = ImagePane(self.split, title="Result")

        self.split.add(self.left_pane, minsize=300)
        self.split.add(self.right_pane, minsize=300)

        # 좌/우 줌 동기화 훅 설치
        self._install_sync_hooks()

        # 단축키(윈도우 기준)
        master.bind("+", lambda e: self._zoom_both("in"))
        master.bind("-", lambda e: self._zoom_both("out"))
        master.bind("<Return>", lambda e: self._fit_both())

        self._update_buttons()

    # ===== 동기화/배치 헬퍼 =====
    def _install_sync_hooks(self):
        def handler(z, origin):
            other = self.right_pane if origin is self.left_pane else self.left_pane
            if abs(other.zoom - z) > 1e-6:
                # 재귀 방지: 잠시 콜백 해제
                cb = other.on_zoom_changed
                other.on_zoom_changed = None
                other.set_zoom(z)
                other.on_zoom_changed = cb
        self.left_pane.on_zoom_changed = handler
        self.right_pane.on_zoom_changed = handler

    def _center_split(self):
        self.split.update_idletasks()
        w = max(1, self.split.winfo_width())
        try:
            self.split.sash_place(0, w // 2, 1)  # 중앙 배치
        except Exception:
            pass

    # ===== Zoom helpers =====
    def _zoom_both(self, direction):
        if direction == "in":
            self.left_pane.zoom_in(0.2)
            self.right_pane.zoom_in(0.2)
        elif direction == "out":
            self.left_pane.zoom_out(0.2)
            self.right_pane.zoom_out(0.2)

    def _fit_both(self):
        """결과 패널이 있으면 그 기준으로 배율 계산, 없으면 원본 기준."""
        self.master.update_idletasks()

        def fit_zoom(pane: ImagePane, margin=8):
            if pane._pil is None:
                return None
            cw = max(1, pane.canvas.winfo_width() - margin)
            ch = max(1, pane.canvas.winfo_height() - margin)
            iw, ih = pane._pil.size
            z = min(cw / iw, ch / ih)
            return max(pane.min_zoom, min(pane.max_zoom, z))

        z = fit_zoom(self.right_pane) if self.right_pane._pil is not None else fit_zoom(self.left_pane)
        if z is not None:
            # 재귀 방지 위해 한쪽씩 설정
            cb_l, cb_r = self.left_pane.on_zoom_changed, self.right_pane.on_zoom_changed
            self.left_pane.on_zoom_changed = None
            self.right_pane.on_zoom_changed = None
            self.left_pane.set_zoom(z)
            self.right_pane.set_zoom(z)
            self.left_pane.on_zoom_changed, self.right_pane.on_zoom_changed = cb_l, cb_r

        self._center_split()

    def _reset_both(self):
        """1:1(무배율)로 통일 + 스플릿 중앙."""
        cb_l, cb_r = self.left_pane.on_zoom_changed, self.right_pane.on_zoom_changed
        self.left_pane.on_zoom_changed = None
        self.right_pane.on_zoom_changed = None
        self.left_pane.reset_zoom()
        self.right_pane.reset_zoom()
        self.left_pane.on_zoom_changed, self.right_pane.on_zoom_changed = cb_l, cb_r
        self._center_split()

    # --- Model load / change ---
    def change_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model (.pt or .engine)",
            filetypes=[("Ultralytics/Onnx/TensorRT", "*.pt;*.onnx;*.engine"), ("All files", "*.*")]
        )
        if not path:
            return
        self.model_path = path
        self.is_engine = str(self.model_path).lower().endswith(".engine")
        self.engine_imgsz = None
        self.model_label.config(text=f"Model: {self.model_path}  (Device: {self.device})")
        self.model = None
        self._update_buttons()

    def save_result(self):
        if self.vis_bgr is None:
            messagebox.showwarning("Warning", "No result to save. Run inference first.")
            return
        initial = os.path.splitext(os.path.basename(self.image_path or "result"))[0] + ".png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=initial,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            ok = cv2.imwrite(path, self.vis_bgr)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
            return
        if ok:
            messagebox.showinfo("Saved", f"Saved: {path}")
        else:
            messagebox.showerror("Error", "Failed to save file.")

    def _ensure_cuda_for_engine(self):
        if self.is_engine and not torch.cuda.is_available():
            messagebox.showwarning(
                "CUDA required for TensorRT",
                "TensorRT .engine requires a CUDA-capable GPU.\n"
                "Please choose a .pt model to run on CPU."
            )
            path = filedialog.askopenfilename(
                title="Select a .pt model for CPU",
                filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")]
            )
            if not path:
                raise RuntimeError("CPU execution requires a .pt model.")
            self.model_path = path
            self.is_engine = False
            self.engine_imgsz = None
            self.model_label.config(text=f"Model: {self.model_path}  (Device: cpu)")
            self.device = "cpu"

    def _detect_engine_imgsz_from_model(self):
        if not self.is_engine or self.model is None:
            return None
        eng_sz = None
        try:
            over = getattr(self.model, "overrides", {}) or {}
            eng_sz = over.get("imgsz") or over.get("img_size")
            if eng_sz is None:
                args = getattr(self.model, "args", {}) or {}
                eng_sz = args.get("imgsz") or args.get("img_size")
        except Exception:
            pass

        if isinstance(eng_sz, (list, tuple, np.ndarray)):
            try:
                eng_sz = int(max(eng_sz))
            except Exception:
                eng_sz = None
        if isinstance(eng_sz, float):
            eng_sz = int(eng_sz)
        if isinstance(eng_sz, int):
            return eng_sz

        try:
            m = re.search(r'_(\d{3,4})\.engine$', os.path.basename(self.model_path), flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    @staticmethod
    def guess_task_from_filename(path: str):
        base = os.path.basename(str(path)).lower()
        if re.search(r'(^|[._-])obd($|[._-])', base) or 'obb' in base or 'oriented' in base or 'rbox' in base:
            return 'obb'
        if re.search(r'(^|[._-])seg($|[._-])', base) or 'segment' in base or 'segm' in base or 'mask' in base:
            return 'segment'
        return None

    def load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._ensure_cuda_for_engine()

        task_hint = self.guess_task_from_filename(self.model_path)
        force_task = task_hint

        try:
            self.model = YOLO(self.model_path, task=force_task) if force_task else YOLO(self.model_path)
        except TypeError:
            self.model = YOLO(self.model_path)

        if force_task in ("segment", "obb"):
            try:
                setattr(self.model, "task", force_task)
            except Exception:
                pass
            for attr in ("overrides", "args"):
                try:
                    d = getattr(self.model, attr, None)
                    if isinstance(d, dict):
                        d["task"] = force_task
                except Exception:
                    pass

        self.names = getattr(self.model, "names", None)
        if not self.names:
            self.names = getattr(getattr(self.model, "model", None), "names", {}) or {}
        if not isinstance(self.names, dict):
            try:
                self.names = dict(self.names)
            except Exception:
                self.names = {}
        if 0 not in self.names:
            self.names[0] = "object"

        if self.is_engine:
            self.engine_imgsz = self._detect_engine_imgsz_from_model() or 320

        self.mode = force_task or getattr(self.model, "task", None) or "detect"

        self.model_label.config(text=f"Model: {self.model_path}  (Device: {self.device})  [Mode: {self.mode}]")

    def _ensure_engine_imgsz_for_predict(self, src_for_predict, imgsz):
        if not self.is_engine:
            return imgsz

        kwargs = dict(
            source=src_for_predict,
            imgsz=imgsz,
            conf=0.01,
            iou=0.99,
            max_det=1,
            verbose=False,
            device=0,
            half=torch.cuda.is_available(),
            task=self.mode,
        )
        if self.mode == "segment":
            kwargs["retina_masks"] = True

        self.model.predict(**kwargs)
        return imgsz

    # --- UI / IO ---
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        self.image_path = path
        self.src_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.src_bgr is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        self.vis_bgr = None

        # 좌: 원본, 우: 플레이스홀더
        self.left_pane.set_title(f"Original  ({os.path.basename(path)})")
        self.left_pane.set_image(self.src_bgr)
        self.right_pane.set_title("Result")
        self.right_pane.set_image(None, placeholder="Run inference to see result")

        # 1:1로 통일 + 중앙 분할
        self._reset_both()
        self._update_buttons()




    def run_inference(self):
        if self.src_bgr is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        def _infer_thread():
            try:
                self.btn_infer.config(state=tk.DISABLED)
                self.btn_save.config(state=tk.DISABLED)
                self.master.config(cursor="watch")

                self.load_model()

                # ── ONNX 감지 ─────────────────────────────────────────────
                is_onnx = str(self.model_path).lower().endswith(".onnx")
                # ─────────────────────────────────────────────────────────

                conf = float(self.conf_var.get())
                iou = float(self.iou_var.get())
                mthr = float(self.maskthr_var.get()) if self.mode == "segment" else None
                imgsz = self.engine_imgsz or (640 if torch.cuda.is_available() else 512)

                src_for_predict = self.src_bgr

                if self.is_engine:
                    imgsz = self._ensure_engine_imgsz_for_predict(src_for_predict, imgsz)

                # ── warmup: ONNX는 생략 (ORT 백엔드라 의미 없고, 장치 혼선 방지) ──
                try:
                    if (not is_onnx) and hasattr(self.model, "warmup"):
                        warmup_device = (0 if (self.is_engine and torch.cuda.is_available())
                                        else (0 if self.device == "cuda" else "cpu"))
                        self.model.warmup(imgsz=(1, 3, imgsz, imgsz), device=warmup_device)
                except Exception:
                    pass

                # CUDA 동기화는 실제 CUDA 사용시에만
                if (not is_onnx) and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                predict_args = dict(
                    source=src_for_predict,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    max_det=(100 if self.mode == "segment" else 50),
                    agnostic_nms=False,
                    verbose=False,
                    task=self.mode,
                )
                if self.mode == "segment":
                    predict_args["retina_masks"] = True

                if self.is_engine:
                    if not torch.cuda.is_available():
                        raise RuntimeError("TensorRT engine requires CUDA (GPU).")
                    predict_args["device"] = 0
                    predict_args["half"] = torch.cuda.is_available()
                else:
                    if is_onnx:
                        # ⚠️ PyTorch CUDA(전처리) 사용을 막기 위해 CPU로 설정
                        #     ONNXRuntime는 내부적으로 CUDAExecutionProvider를 사용해 GPU에서 추론합니다.
                        predict_args["device"] = "cpu"
                        predict_args["half"] = False  # ORT에는 half 의미 없음
                    else:
                        predict_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
                        predict_args["half"] = torch.cuda.is_available()

                results = self.model.predict(**predict_args)

                if (not is_onnx) and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                if not results:
                    messagebox.showinfo("Result", "No results.")
                    return

                res = results[0]

                if self.is_engine and self.mode == "segment":
                    if (getattr(res, "masks", None) is None) or (getattr(getattr(res, "masks", None), "data", None) is None):
                        messagebox.showerror(
                            "Engine is not SEG",
                            "This TensorRT engine did not return segmentation masks.\n\n"
                            "It was likely exported as DET/OBB.\n"
                            "Please re-export from a segmentation (.pt) model.\n\n"
                            "Tip) YOLO('your_seg.pt', task='segment').export(format='engine', imgsz=640, half=True, device=0)"
                        )
                        return
                if self.is_engine and self.mode == "obb":
                    if getattr(res, "obb", None) is None:
                        messagebox.showerror(
                            "Engine is not OBB",
                            "This TensorRT engine did not return OBB outputs.\n\n"
                            "It was likely exported as DET.\n"
                            "Please re-export from an OBB .pt model.\n\n"
                            "Tip) YOLO('your_obb.pt', task='obb').export(format='engine', imgsz=640, half=True, device=0)"
                        )
                        return

                safe_names = {}
                if isinstance(self.names, dict):
                    safe_names.update(self.names)
                try:
                    rn = getattr(res, "names", None)
                    if isinstance(rn, dict):
                        safe_names.update(rn)
                except Exception:
                    pass
                try:
                    res.names = safe_names
                except Exception:
                    pass

                base = getattr(res, "orig_img", None)
                if base is None:
                    base = self.src_bgr

                if self.mode == "segment":
                    vis = draw_segmentation_on_image(
                        base, res, safe_names,
                        thr=(mthr if mthr is not None else 0.5),
                        alpha=ALPHA_DEFAULT,
                        line_thickness=LINE_THICKNESS,
                        font_scale=FONT_SCALE,
                        padding=PADDING
                    )
                elif self.mode == "obb":
                    vis = draw_obb_on_image(
                        base, res, safe_names,
                        line_thickness=LINE_THICKNESS,
                        font_scale=FONT_SCALE,
                        padding=PADDING
                    )
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")

                self.vis_bgr = vis

                # 좌/우 업데이트
                self.left_pane.set_title(f"Original  ({os.path.basename(self.image_path)})")
                self.left_pane.set_image(self.src_bgr)
                self.right_pane.set_title(f"Result  [{self.mode.upper()}]")
                self.right_pane.set_image(self.vis_bgr)

                # 1:1 동일 배율 + 중앙 분할
                self._reset_both()

                self.btn_save.config(state=tk.NORMAL)

                total_ms = (t1 - t0) * 1000.0
                device_str = (
                    predict_args.get("device", "cpu") if is_onnx
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
                print(
                    f"[INFO] {self.mode.upper()} imgsz={imgsz} device={device_str} "
                    f"engine={self.is_engine} time={total_ms:.1f}ms"
                )

            except Exception as e:
                import traceback as tb
                print("[ERROR] Exception:", e, "\n", tb.format_exc())
                messagebox.showerror("Error", f"Inference failed:\n{e}")
            finally:
                self.master.config(cursor="")
                self._update_buttons()

        threading.Thread(target=_infer_thread, daemon=True).start()











    def clear_view(self):
        self.image_path = None
        self.src_bgr = None
        self.vis_bgr = None

        self.left_pane.set_title("Original")
        self.left_pane.set_image(None, placeholder="Load an image to begin")
        self.right_pane.set_title("Result")
        self.right_pane.set_image(None, placeholder="Run inference to see result")

        self._update_buttons()

    def _update_buttons(self):
        self.btn_infer.config(state=(tk.NORMAL if self.src_bgr is not None else tk.DISABLED))
        self.btn_save.config(state=(tk.NORMAL if self.vis_bgr is not None else tk.DISABLED))

def main():
    root = tk.Tk()
    app = YoloApp(root)
    app.clear_view()
    root.geometry("1200x800")
    root.minsize(900, 600)
    root.mainloop()

if __name__ == "__main__":
    main()