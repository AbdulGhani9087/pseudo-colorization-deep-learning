"""
Modernized Tkinter GUI for exploring pseudo-color mappings on grayscale images.
Focuses on LUT-based colorization plus level slicing, providing quick controls
for palette, gamma, SE boost, and threshold tuning inside a sleek layout.
"""
from __future__ import annotations

import os
from typing import Dict

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

# --- Import local modules ---
try:
    import baselines
    from model import triang_infer
except ImportError as e:
    messagebox.showerror(
        "Import Error",
        f"Failed to import local modules: {e}. Ensure all .py files are present.",
    )
    raise

# Paths / constants
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_TRIANG_WEIGHTS = os.path.join(PROJ_ROOT, "results", "triang_model.pth")

# Constants for consistent display size
PREVIEW_SIZE = 256  # Size for preview squares


class PseudoColorGUI:
    """Single-window experience with sidebar controls + gallery cards."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pseudo-Colorization Studio")
        self.root.configure(bg="#10121b")

        self.img_gray: np.ndarray | None = None
        self.img_results: Dict[str, np.ndarray] = {}
        self.img_original_tk: ImageTk.PhotoImage | None = None
        self.img_result_tks: Dict[str, ImageTk.PhotoImage] = {}

        self.method_vars: Dict[str, tk.BooleanVar] = {}
        self.th1_var = tk.IntVar(value=85)
        self.th2_var = tk.IntVar(value=170)
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.cmap_var = tk.StringVar(value="viridis")
        self.se_lut_var = tk.BooleanVar(value=False)
        self.triang_saturation_var = tk.DoubleVar(value=2.5)
        self.triang_intensity_var = tk.DoubleVar(value=1.8)
        self.triang_lut_enable_var = tk.BooleanVar(value=True)
        self.triang_lut_cmap_var = tk.StringVar(value="viridis")
        self.triang_lut_gamma_var = tk.DoubleVar(value=1.0)
        self.triang_lut_blend_var = tk.DoubleVar(value=0.6)
        self.status_var = tk.StringVar(value="Load a grayscale photo to begin.")
        self.slice_info_var = tk.StringVar(value=self._format_slice_ranges())

        self.method_labels = {
            "Original": None,
            "LUT (Viridis)": None,
            "Level-Slice": None,
            "Triang Model": None,
        }

        self._create_widgets()

    # --------------------------------------------------------------------- UI
    def _create_widgets(self):
        main = tk.Frame(self.root, bg="#10121b")
        main.pack(fill="both", expand=True, padx=12, pady=12)

        # Scrollable controls sidebar setup (canvas + vertical scrollbar + frame)
        canvas = tk.Canvas(main, bg="#10121b", highlightthickness=0, width=280)
        canvas.pack(side="left", fill="y", expand=False)

        v_scroll = tk.Scrollbar(main, orient="vertical", command=canvas.yview)
        v_scroll.pack(side="left", fill="y")

        canvas.configure(yscrollcommand=v_scroll.set)

        control_frame = tk.Frame(canvas, bg="#16182b", bd=0)
        canvas.create_window((0, 0), window=control_frame, anchor="nw")

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        control_frame.bind("<Configure>", on_configure)

        # --- Add all controls to control_frame as before ---
        tk.Label(
            control_frame,
            text="Pseudo-Color Studio",
            font=("Segoe UI", 14, "bold"),
            fg="#f6f7fb",
            bg="#16182b",
        ).pack(padx=18, pady=(18, 8))

        tk.Label(
            control_frame,
            text="Upload a grayscale image and explore colorful mappings.",
            wraplength=220,
            justify="left",
            fg="#a8acc5",
            bg="#16182b",
        ).pack(padx=18, pady=(0, 18))

        self._styled_button(control_frame, "Upload Photo", self.load_image, "#4CAF50").pack(
            fill="x", padx=18, pady=(0, 10)
        )
        self._styled_button(
            control_frame,
            "Process Selected",
            lambda: self.process_all_methods(silent=False),
            "#008CBA",
        ).pack(fill="x", padx=18, pady=(0, 10))
        self._styled_button(control_frame, "Save Results", self.save_all_results, "#f44336").pack(
            fill="x", padx=18, pady=(0, 18)
        )

        self._build_method_section(control_frame)
        self._build_lut_section(control_frame)
        self._build_slice_section(control_frame)
        self._build_triang_section(control_frame)

        # Right side visualization area
        viz_frame = tk.Frame(main, bg="#10121b")
        viz_frame.pack(side="left", fill="both", expand=True)

        grid = tk.Frame(viz_frame, bg="#10121b")
        grid.pack(fill="both", expand=True)

        methods = list(self.method_labels.keys())
        for i, method in enumerate(methods):
            label = self._create_card(grid, title=method, row=i // 2, column=i % 2)
            self.method_labels[method] = label
            self._show_placeholder(method, label)

        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            bg="#0d0f16",
            fg="#bfc3dd",
            padx=16,
            pady=6,
        )
        status.pack(fill="x", side="bottom")

    def _styled_button(self, parent: tk.Widget, text: str, command, color: str) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            activebackground=color,
            relief="flat",
            padx=8,
            pady=6,
        )

    def _section_frame(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.LabelFrame(
            parent,
            text=title,
            bg="#1c1f36",
            fg="#8f95c4",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
            labelanchor="n",
        )
        frame.pack(fill="x", padx=18, pady=(0, 16))
        return frame

    def _build_method_section(self, parent: tk.Widget):
        section = self._section_frame(parent, "Outputs")
        for name, default in [
            ("LUT (Viridis)", True),
            ("Level-Slice", True),
            ("Triang Model", False),
        ]:
            var = tk.BooleanVar(value=default)
            chk = tk.Checkbutton(
                section,
                text=name,
                variable=var,
                bg=section["bg"],
                fg="#e5e9ff",
                selectcolor="#2d3253",
                activebackground=section["bg"],
            )
            chk.pack(anchor="w", pady=4)
            self.method_vars[name] = var

    def _build_lut_section(self, parent: tk.Widget):
        section = self._section_frame(parent, "LUT Styling")

        tk.Label(section, text="Colormap", bg=section["bg"], fg="#c8cbe0").pack(anchor="w")
        ttk.Combobox(
            section,
            textvariable=self.cmap_var,
            values=["viridis", "magma", "plasma", "inferno", "cividis", "turbo"],
            state="readonly",
        ).pack(fill="x", pady=4)

        tk.Label(section, text="Gamma", bg=section["bg"], fg="#c8cbe0").pack(anchor="w", pady=(8, 0))
        tk.Scale(
            section,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.gamma_var,
            bg=section["bg"],
            fg="#e5e9ff",
            troughcolor="#2d3253",
            highlightthickness=0,
        ).pack(fill="x")

        tk.Checkbutton(
            section,
            text="SE boost",
            variable=self.se_lut_var,
            bg=section["bg"],
            fg="#e5e9ff",
            selectcolor="#2d3253",
            activebackground=section["bg"],
        ).pack(anchor="w", pady=(8, 0))

    def _build_slice_section(self, parent: tk.Widget):
        section = self._section_frame(parent, "Level Slice")

        tk.Label(section, text="Threshold 1", bg=section["bg"], fg="#c8cbe0").pack(anchor="w")
        tk.Scale(
            section,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self.th1_var,
            command=self._on_threshold_change,
            bg=section["bg"],
            fg="#e5e9ff",
            troughcolor="#2d3253",
            highlightthickness=0,
        ).pack(fill="x")

        tk.Label(section, text="Threshold 2", bg=section["bg"], fg="#c8cbe0").pack(anchor="w", pady=(8, 0))
        tk.Scale(
            section,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self.th2_var,
            command=self._on_threshold_change,
            bg=section["bg"],
            fg="#e5e9ff",
            troughcolor="#2d3253",
            highlightthickness=0,
        ).pack(fill="x")
        tk.Label(
            section,
            textvariable=self.slice_info_var,
            bg=section["bg"],
            fg="#9ea3cf",
            anchor="w",
            justify="left",
            wraplength=220,
            pady=6,
        ).pack(fill="x")

    def _build_triang_section(self, parent: tk.Widget):
        section = self._section_frame(parent, "Triang Model")

        tk.Label(section, text="Saturation", bg=section["bg"], fg="#c8cbe0").pack(anchor="w")
        tk.Scale(
            section,
            from_=0.5,
            to=5.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.triang_saturation_var,
            bg=section["bg"],
            fg="#e5e9ff",
            troughcolor="#2d3253",
            highlightthickness=0,
        ).pack(fill="x")

        tk.Label(section, text="Intensity", bg=section["bg"], fg="#c8cbe0").pack(anchor="w", pady=(8, 0))
        tk.Scale(
            section,
            from_=0.5,
            to=4.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.triang_intensity_var,
            bg=section["bg"],
            fg="#e5e9ff",
            troughcolor="#2d3253",
            highlightthickness=0,
        ).pack(fill="x")

        # LUT Enhancement Section
        tk.Label(section, text="LUT Enhancement", bg=section["bg"], fg="#c8cbe0", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(12, 4))
        
        tk.Checkbutton(
            section,
            text="Apply LUT Colormap",
            variable=self.triang_lut_enable_var,
            bg=section["bg"],
            fg="#e5e9ff",
            selectcolor="#2d3253",
            activebackground=section["bg"],
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(section, text="Colormap", bg=section["bg"], fg="#c8cbe0").pack(anchor="w", pady=(4, 0))
        ttk.Combobox(
            section,
            textvariable=self.triang_lut_cmap_var,
            values=["viridis", "magma", "plasma", "inferno", "cividis", "turbo", "hot", "cool", "spring", "summer"],
            state="readonly",
            width=18,
        ).pack(fill="x", pady=4)

        tk.Label(section, text="LUT Gamma", bg=section["bg"], fg="#c8cbe0").pack(anchor="w", pady=(4, 0))
        tk.Scale(
            section,
            from_=0.3,
            to=2.5,
            resolution=0.1,
            orient="horizontal",
            variable=self.triang_lut_gamma_var,
            bg=section["bg"],
            fg="#e5e9ff",
            troughcolor="#2d3253",
            highlightthickness=0,
        ).pack(fill="x")

        tk.Label(section, text="LUT Blend", bg=section["bg"], fg="#c8cbe0").pack(anchor="w", pady=(8, 0))
        tk.Scale(
            section,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.triang_lut_blend_var,
            bg=section["bg"],
            fg="#e5e9ff",
            troughcolor="#2d3253",
            highlightthickness=0,
        ).pack(fill="x")

    def _create_card(self, parent: tk.Frame, title: str, row: int, column: int) -> tk.Label:
        card = tk.Frame(parent, bg="#181b2f", highlightbackground="#272c47", highlightthickness=1)
        card.grid(row=row, column=column, padx=14, pady=14, sticky="nsew")
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(column, weight=1)

        tk.Label(card, text=title, font=("Segoe UI", 11, "bold"), fg="#f6f7fb", bg=card["bg"]).pack(
            pady=(12, 6)
        )
        label = tk.Label(
            card,
            width=PREVIEW_SIZE,
            height=PREVIEW_SIZE,
            bg="#05060b",
            bd=0,
        )
        label.pack(padx=10, pady=(0, 6))

        if title == "Level-Slice":
            tk.Label(
                card,
                textvariable=self.slice_info_var,
                font=("Segoe UI", 9),
                fg="#b3b7d6",
                bg=card["bg"],
                wraplength=PREVIEW_SIZE,
                justify="center",
            ).pack(padx=10, pady=(0, 10))
        else:
            tk.Label(card, text="", bg=card["bg"]).pack(pady=(0, 10))
        return label

    def _show_placeholder(self, title: str, label: tk.Label):
        img = Image.new("RGB", (PREVIEW_SIZE, PREVIEW_SIZE), color="#05060b")
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font = ImageFont.load_default()
            text = f"{title}\n\nNo Image Loaded" if title == "Original" else f"{title}\n\nReady"
            draw.text((14, PREVIEW_SIZE // 2 - 20), text, fill="#f6f7fb", font=font)
        except Exception:
            pass

        tk_img = ImageTk.PhotoImage(img)
        self.img_result_tks[title] = tk_img
        label.config(image=tk_img, width=PREVIEW_SIZE, height=PREVIEW_SIZE)
        label.image = tk_img

    def _resize_and_convert_to_tk(self, arr: np.ndarray) -> ImageTk.PhotoImage:
        if arr.ndim == 2:
            img = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3:
            img = Image.fromarray(arr)
        else:
            raise ValueError("Input array must be 2D or 3D")
        img = img.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    # ----------------------------------------------------------------- Actions
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff")]
        )
        if not path:
            return

        try:
            img = Image.open(path).convert("L")
            self.img_gray = np.array(img)
            self.img_results = {}

            self.img_original_tk = self._resize_and_convert_to_tk(self.img_gray)
            orig_label = self.method_labels["Original"]
            orig_label.config(image=self.img_original_tk, text="")
            orig_label.image = self.img_original_tk

            for name, label in self.method_labels.items():
                if name != "Original":
                    self._show_placeholder(name, label)

            messagebox.showinfo("Success", f"Image loaded successfully: {os.path.basename(path)}")
            self._update_status(f"Loaded {os.path.basename(path)}. Ready to process.")

            try:
                self.process_all_methods(silent=True)
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load image: {exc}")
            self.img_gray = None
            self._update_status("Image load failed. Please try another file.")

    def process_all_methods(self, *, silent: bool = False):
        if self.img_gray is None:
            messagebox.showinfo("Info", "Please load a grayscale image first.")
            self._update_status("Upload a grayscale image to start processing.")
            return

        gray = self.img_gray
        self.img_results = {}

        try:
            if self.method_vars.get("LUT (Viridis)", tk.BooleanVar()).get():
                se_flag = bool(self.se_lut_var.get())
                cmap = self.cmap_var.get() or "viridis"
                gamma = max(0.1, float(self.gamma_var.get()))
                self.img_results["LUT (Viridis)"] = baselines.apply_lut(
                    gray, cmap=cmap, se=se_flag, gamma=gamma
                )

            if self.method_vars.get("Level-Slice", tk.BooleanVar()).get():
                t1 = int(self.th1_var.get())
                t2 = int(self.th2_var.get())
                if t1 > t2:
                    t1, t2 = t2, t1
                thresholds = [t1, t2]
                colors = [(0, 0, 128), (0, 128, 0), (128, 0, 0)]
                self.img_results["Level-Slice"] = baselines.level_slice(gray, thresholds, colors)

            if self.method_vars.get("Triang Model", tk.BooleanVar()).get():
                try:
                    if not os.path.exists(DEFAULT_TRIANG_WEIGHTS):
                        raise FileNotFoundError(
                            f"Triang weights not found at {DEFAULT_TRIANG_WEIGHTS}. "
                            "Train the model with src/train.py or update the path in gui.py."
                        )
                    saturation = float(self.triang_saturation_var.get())
                    intensity = float(self.triang_intensity_var.get())
                    lut_enable = bool(self.triang_lut_enable_var.get())
                    lut_cmap = self.triang_lut_cmap_var.get() if lut_enable else None
                    lut_gamma = float(self.triang_lut_gamma_var.get())
                    lut_blend = float(self.triang_lut_blend_var.get())
                    self.img_results["Triang Model"] = triang_infer(
                        gray,
                        weights_path=DEFAULT_TRIANG_WEIGHTS,
                        saturation=saturation,
                        intensity=intensity,
                        apply_lut=lut_enable,
                        lut_cmap=lut_cmap,
                        lut_gamma=lut_gamma,
                        lut_blend=lut_blend,
                    )
                except Exception as exc:
                    if not silent:
                        messagebox.showerror("Triang Error", f"Triang model failed: {exc}")

            for name, label in self.method_labels.items():
                if name == "Original":
                    continue
                arr = self.img_results.get(name)
                if arr is not None:
                    tk_img = self._resize_and_convert_to_tk(arr)
                    self.img_result_tks[name] = tk_img
                    label.config(image=tk_img, text="")
                    label.image = tk_img
                else:
                    self._show_placeholder(name, label)

            if not silent:
                messagebox.showinfo("Success", "Selected pseudo-color methods processed.")
            self._update_status(f"Generated {len(self.img_results)} view(s). Save to share!")
        except Exception as exc:
            messagebox.showerror("Error", f"Processing failed: {exc}")
            print(f"Detailed error: {exc}")
            self.img_results = {}
            self._update_status("Processing failed. Check console for details.")

    def save_all_results(self):
        if not self.img_results:
            messagebox.showinfo("Info", "No results to save. Process methods first.")
            self._update_status("Nothing to save yet. Run processing first.")
            return

        save_dir = filedialog.askdirectory(title="Select Folder to Save Results")
        if not save_dir:
            return

        try:
            for name, arr in self.img_results.items():
                filename = (
                    name.lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("(", "")
                    .replace(")", "")
                )
                path = os.path.join(save_dir, f"result_{filename}.png")
                Image.fromarray(arr).save(path)

            messagebox.showinfo("Success", f"All {len(self.img_results)} results saved to:\n{save_dir}")
            self._update_status(f"Saved outputs to {save_dir}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save results: {exc}")
            self._update_status("Save failed. Try a different folder.")

    # ----------------------------------------------------------------- Helpers
    def _update_status(self, text: str):
        self.status_var.set(text)

    def _format_slice_ranges(self) -> str:
        t1 = int(self.th1_var.get())
        t2 = int(self.th2_var.get())
        if t1 > t2:
            t1, t2 = t2, t1
        return f"Ranges: 0-{t1} | {t1}-{t2} | {t2}+"

    def _on_threshold_change(self, _value=None):
        self.slice_info_var.set(self._format_slice_ranges())
        if self.img_gray is None:
            return
        if not self.method_vars.get("Level-Slice", tk.BooleanVar(value=True)).get():
            return
        self._update_status(f"Thresholds updated -> {self.slice_info_var.get()}")
        self.process_all_methods(silent=True)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x700")  # Set window size big enough for scrolling + preview
    app = PseudoColorGUI(root)
    root.minsize(600, 500)
    root.mainloop()
