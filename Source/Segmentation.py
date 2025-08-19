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

try:
    import torch
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        f"[ImportError] {e}\n"
        "Install deps:\n"
        "    pip install ultralytics opencv-python pillow torch --upgrade"
    )

# ====== CONFIG ======
# 필요 시 바꿔 써: 세그 엔진 기본 경로 예시
DEFAULT_MODEL_PATH = r"D:\TOYPROJECT\OS\Model\SEG\SEG.engine"  # or .pt
CONF_DEFAULT = 0.25
IOU_DEFAULT = 0.45
LINE_THICKNESS = 2
FONT_SCALE = 0.6
PADDING = 6
MASK_THR_DEFAULT = 0.5
ALPHA_DEFAULT = 0.45
WINDOW_TITLE = "YOLOv11 — SEGMENTATION only"
# =====================


# ---------- SEG overlay ----------
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
    """
    result.masks 기반으로 폴리곤(외곽선) + 반투명 채우기.
    - 채우기: binary mask 기반 addWeighted
    - 라벨: 최대 외곽 컨투어 중심 근처에 캡션
    - cls/conf: result.boxes의 cls/conf 사용(있으면)
    """
    base = img_bgr.copy()
    h, w = base.shape[:2]

    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        cv2.putText(
            base,
            "No segmentation masks",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return base

    # N x H x W
    mask_data = masks.data
    mask_np = mask_data.cpu().numpy() if hasattr(mask_data, "cpu") else np.asarray(mask_data)

    # 클래스/점수는 boxes에서 끌어옴(세그 결과에도 같이 옴)
    boxes = getattr(result, "boxes", None)
    cls_ids = None
    confs = None
    if boxes is not None:
        if getattr(boxes, "cls", None) is not None:
            cls_ids = boxes.cls.int().cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.astype(int)
        if getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "cpu") else boxes.conf

    N = mask_np.shape[0]
    # 정렬(위->아래, 좌->우) 위해 중심 추정
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
        m = (mask_np[i] > thr).astype(np.uint8)  # 0/1
        if m.max() == 0:
            continue

        # 색상은 class(seed) 기반
        cls_i = int(cls_ids[i]) if (isinstance(cls_ids, np.ndarray) and i < len(cls_ids)) else i
        rng = np.random.default_rng(seed=int(cls_i * 123457))
        color = rng.integers(low=64, high=255, size=3, dtype=np.uint8).tolist()

        # 채우기(반투명)
        colored = np.zeros_like(overlay)
        colored[m == 1] = color
        overlay = cv2.add(overlay, colored)

        # 외곽선 폴리곤 그리기 (컨투어 이용)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 가장 큰 외곽선 기준으로 캡션 위치 계산
            c = max(contours, key=cv2.contourArea)
            cv2.polylines(base, [c], isClosed=True, color=tuple(int(x) for x in color), thickness=line_thickness)

            M = cv2.moments(c)
            if M["m00"] > 0:
                tx = int(M["m10"] / M["m00"])
                ty = int(M["m01"] / M["m00"])
            else:
                x, y, ww, hh = cv2.boundingRect(c)
                tx, ty = x + ww // 2, y + hh // 2

            # 라벨/점수
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

    # 반투명 블렌딩
    base = cv2.addWeighted(base, 1.0, overlay, alpha, 0)
    return base


class SEGOnlyApp:
    def __init__(self, master):
        self.master = master
        master.title(WINDOW_TITLE)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = DEFAULT_MODEL_PATH
        self.model = None

        # engine flags
        self.is_engine = str(self.model_path).lower().endswith(".engine")
        self.engine_imgsz = None  # 엔진 입력 크기 (로드 후 보정)

        # state
        self.image_path = None
        self.src_bgr = None
        self.vis_bgr = None
        self.names = {}

        # ========== UI ==========
        top = tk.Frame(master)
        top.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.model_label = tk.Label(
            top, text=f"Model: {self.model_path}  (Device: {self.device})", anchor="w"
        )
        self.model_label.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

        self.btn_select_model = tk.Button(top, text="Change Model...", command=self.change_model)
        self.btn_select_model.pack(side=tk.RIGHT)

        actions = tk.Frame(master)
        actions.pack(fill=tk.X, padx=8, pady=(4, 2))

        self.btn_load = tk.Button(actions, text="Load Image", width=14, command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=4)

        self.btn_infer = tk.Button(
            actions, text="Run Inference (SEG)", width=18, command=self.run_inference, state=tk.DISABLED
        )
        self.btn_infer.pack(side=tk.LEFT, padx=4)

        self.btn_save = tk.Button(
            actions, text="Save Result", width=14, command=self.save_result, state=tk.DISABLED
        )
        self.btn_save.pack(side=tk.LEFT, padx=4)

        self.btn_clear = tk.Button(actions, text="Clear", width=10, command=self.clear_view, state=tk.NORMAL)
        self.btn_clear.pack(side=tk.LEFT, padx=4)

        params = tk.Frame(master)
        params.pack(fill=tk.X, padx=8, pady=(0, 8))

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

        self.canvas = tk.Label(master, bd=1, relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        # ========================

        self._update_buttons()

    # --- Model load / change ---
    def change_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model (.pt or .engine)",
            filetypes=[("Ultralytics/TensorRT", "*.pt;*.engine"), ("All files", "*.*")]
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
        initial = os.path.splitext(os.path.basename(self.image_path or "result"))[0] + "_seg.png"
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
        """엔진 입력 크기를 overrides/args/파일명에서 최대한 추출"""
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

        # 파일명 힌트 *_320.engine
        try:
            m = re.search(r'_(\d{3,4})\.engine$', os.path.basename(self.model_path), flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    def load_model(self):
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._ensure_cuda_for_engine()

        # 엔진일 때는 task='segment' 강제(메타 부재 대비)
        if self.is_engine:
            try:
                self.model = YOLO(self.model_path, task="segment")
            except TypeError:
                self.model = YOLO(self.model_path)
            try:
                setattr(self.model, "task", "segment")
                if hasattr(self.model, "overrides") and isinstance(self.model.overrides, dict):
                    self.model.overrides["task"] = "segment"
                if hasattr(self.model, "args") and isinstance(self.model.args, dict):
                    self.model.args["task"] = "segment"
            except Exception:
                pass
        else:
            self.model = YOLO(self.model_path)

        # names 보정
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

        # 엔진 입력 크기 감지
        if self.is_engine:
            self.engine_imgsz = self._detect_engine_imgsz_from_model()
            if not self.engine_imgsz:
                self.engine_imgsz = 320  # 기본 후보

        self.model_label.config(
            text=f"Model: {self.model_path}  (Device: {self.device})  [Mode: SEG]"
        )

    def _ensure_engine_imgsz_for_predict(self, src_for_predict, imgsz):
        """엔진 입력 크기가 틀리면 에러 메시지를 파싱해 자동 교정한다."""
        if not self.is_engine:
            return imgsz
        try:
            # 가벼운 probe (반드시 task='segment')
            self.model.predict(
                source=src_for_predict,
                imgsz=imgsz,
                conf=0.01,
                iou=0.99,
                max_det=1,
                verbose=False,
                device=0,
                half=torch.cuda.is_available(),
                task="segment",
                retina_masks=True,
            )
            return imgsz  # 성공
        except Exception as e:
            msg = str(e)
            m = re.search(r'max model size\s*\(1,\s*3,\s*(\d{3,4}),\s*\1\)', msg)
            if m:
                fix = int(m.group(1))
                print(f"[ENGINE] correcting imgsz {imgsz} -> {fix} from error message")
                self.engine_imgsz = fix
                return fix
            # 마지막 수단: 320/640 후보 시도
            for cand in (320, 640):
                if cand == imgsz:
                    continue
                try:
                    self.model.predict(
                        source=src_for_predict,
                        imgsz=cand,
                        conf=0.01,
                        iou=0.99,
                        max_det=1,
                        verbose=False,
                        device=0,
                        half=torch.cuda.is_available(),
                        task="segment",
                        retina_masks=True,
                    )
                    print(f"[ENGINE] imgsz auto-selected {cand}")
                    self.engine_imgsz = cand
                    return cand
                except Exception:
                    pass
            raise

    # --- UI / IO ---
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.image_path = path
        self.src_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.src_bgr is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        self.vis_bgr = None
        self._update_view(self.src_bgr)
        self._update_buttons()

    def _update_view(self, img_bgr):
        display = self._fit_to_window(img_bgr, max_w=1280, max_h=720)
        img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

    @staticmethod
    def _fit_to_window(img_bgr, max_w=1280, max_h=720):
        h, w = img_bgr.shape[:2]
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_bgr

    def run_inference(self):
        if self.src_bgr is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        def worker():
            try:
                self.btn_infer.config(state=tk.DISABLED)
                self.btn_save.config(state=tk.DISABLED)
                self.master.config(cursor="watch")

                self.load_model()

                conf = float(self.conf_var.get())
                iou = float(self.iou_var.get())
                mthr = float(self.maskthr_var.get())
                imgsz = self.engine_imgsz or (640 if torch.cuda.is_available() else 512)

                # 항상 numpy 입력으로 통일 → 좌표계/스케일 mismatch 방지
                src_for_predict = self.src_bgr

                # 엔진이면 imgsz 자동 교정
                if self.is_engine:
                    imgsz = self._ensure_engine_imgsz_for_predict(src_for_predict, imgsz)

                # optional warmup
                try:
                    if hasattr(self.model, "warmup"):
                        self.model.warmup(
                            imgsz=(1, 3, imgsz, imgsz),
                            device=(0 if (self.is_engine and torch.cuda.is_available())
                                    else (0 if self.device == 'cuda' else 'cpu'))
                        )
                except Exception:
                    pass

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                predict_args = dict(
                    source=src_for_predict,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    max_det=100,
                    agnostic_nms=False,
                    verbose=False,
                    task="segment",       # ★ 항상 segmentation
                    retina_masks=True,    # 고해상도 마스크
                )
                if self.is_engine:
                    if not torch.cuda.is_available():
                        raise RuntimeError("TensorRT engine requires CUDA (GPU).")
                    predict_args["device"] = 0
                    predict_args["half"] = torch.cuda.is_available()
                else:
                    predict_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
                    predict_args["half"] = torch.cuda.is_available()

                results = self.model.predict(**predict_args)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                if not results:
                    messagebox.showinfo("Result", "No results.")
                    return

                res = results[0]

                # ★ 엔진인데 마스크가 없으면 세그 엔진 아님
                if self.is_engine and (getattr(res, "masks", None) is None or getattr(res.masks, "data", None) is None):
                    messagebox.showerror(
                        "Engine is not SEG",
                        "This TensorRT engine did not return segmentation masks.\n\n"
                        "It was likely exported as DET/OBB.\n"
                        "Please re-export from a segmentation (.pt) model.\n\n"
                        "Tip) YOLO('your_seg.pt', task='segment').export(format='engine', imgsz=640, half=True, device=0)"
                    )
                    return

                # names 보정
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

                # 항상 res.orig_img 기준으로 그리기 → 완전 정렬
                base = getattr(res, "orig_img", None)
                if base is None:
                    base = self.src_bgr

                vis = draw_segmentation_on_image(
                    base, res, safe_names, thr=mthr, alpha=ALPHA_DEFAULT,
                    line_thickness=LINE_THICKNESS, font_scale=FONT_SCALE, padding=PADDING
                )

                self.vis_bgr = vis
                self._update_view(self.vis_bgr)
                self.btn_save.config(state=tk.NORMAL)

                total_ms = (t1 - t0) * 1000.0
                print(
                    f"[INFO] SEG-only imgsz={imgsz} device={'cuda' if torch.cuda.is_available() else 'cpu'} "
                    f"engine={self.is_engine} time={total_ms:.1f}ms"
                )

            except Exception as e:
                import traceback as tb
                print("[ERROR] Exception:", e, "\n", tb.format_exc())
                messagebox.showerror("Error", f"Inference failed:\n{e}")
            finally:
                self.master.config(cursor="")
                self._update_buttons()

        threading.Thread(target=worker, daemon=True).start()

    def clear_view(self):
        self.image_path = None
        self.src_bgr = None
        self.vis_bgr = None
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "Load an image to begin",
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        self._update_view(blank)
        self._update_buttons()

    def _update_buttons(self):
        self.btn_infer.config(state=(tk.NORMAL if self.src_bgr is not None else tk.DISABLED))
        self.btn_save.config(state=(tk.NORMAL if self.vis_bgr is not None else tk.DISABLED))


def main():
    root = tk.Tk()
    app = SEGOnlyApp(root)
    app.clear_view()
    root.geometry("1000x700")
    root.mainloop()


if __name__ == "__main__":
    main()