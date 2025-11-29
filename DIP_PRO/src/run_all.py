"""Simple test runner to exercise the project's modules.

Saves outputs into `results/` and prints basic metrics.
Run from project root with the venv python:
  .\.venv\Scripts\python.exe src\run_all.py
"""
import os
import sys
import traceback
import argparse

from PIL import Image
import numpy as np

# Ensure we can import modules from src/ by adding src directory to path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(HERE, '..'))
SRC_DIR = os.path.join(PROJ_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import baselines


def save_arr_as_png(arr, path):
    Image.fromarray(arr).save(path)


def normalize_for_vis(arr):
    a = arr.astype(np.float32)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return (a * 255).astype(np.uint8)


def main():
    try:
        parser = argparse.ArgumentParser(description='Run demo pipeline or open GUI.')
        parser.add_argument('--gui', action='store_true', help='Open GUI to display results instead of saving files')
        parser.add_argument('--preload', action='store_true', help='When used with --gui, preload example image and auto-process')
        args = parser.parse_args()
        data_dir = os.path.join(PROJ_ROOT, 'data')
        results_dir = os.path.join(PROJ_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)

        img_path = os.path.join(data_dir, 'example_gray.png')
        if not os.path.exists(img_path):
            # fallback: create a gradient
            arr = np.tile(np.arange(256, dtype=np.uint8), (256,1))
            Image.fromarray(arr).save(img_path)

        gray = np.array(Image.open(img_path).convert('L'))

        # If GUI requested, launch the Tkinter GUI. By default do NOT preload
        # the example image so the user must select a photo via the UI. Use
        # --preload with --gui to restore the previous auto-load behavior.
        if args.gui:
            try:
                import tkinter as tk
                import gui as gui_mod

                root = tk.Tk()
                app = gui_mod.PseudoColorGUI(root)

                if args.preload:
                    # Provide the example image to the GUI and trigger processing
                    app.img_gray = gray
                    app.img_original_tk = app._resize_and_convert_to_tk(gray)
                    orig_label = app.method_labels.get('Original')
                    if orig_label is not None:
                        orig_label.config(image=app.img_original_tk, text='')
                        orig_label.image = app.img_original_tk

                    # Run the processing so panels are populated
                    app.process_all_methods(silent=True)

                root.minsize(850, 600)
                root.mainloop()
                return
            except Exception:
                print('Failed to launch GUI:')
                traceback.print_exc()

        # Run baselines: only LUT and Level-Slice
        lut = baselines.apply_lut(gray, cmap='viridis')
        save_arr_as_png(lut, os.path.join(results_dir, 'run_lut.png'))

        th = [85, 170]
        cols = [(0,0,128),(0,128,0),(128,0,0)]
        lvl = baselines.level_slice(gray, th, cols)
        save_arr_as_png(lvl, os.path.join(results_dir, 'run_level.png'))

        print('Saved LUT and Level-Slice outputs to', results_dir)

    except Exception:
        print('Error during run:')
        traceback.print_exc()


if __name__ == '__main__':
    main()
