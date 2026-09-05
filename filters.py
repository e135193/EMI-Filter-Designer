#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CE102 EMI Power-Line Filter Designer
=====================================

MIL-STD-461F/G CE102 (Conducted Emissions, Power Leads, 10 kHz - 10 MHz)
interactive filter design tool.
"""

import os
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

warnings.filterwarnings("ignore", message="Attempt to set non-positive xlim")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMATIC_PATH = os.path.join(SCRIPT_DIR, "SCH.png")

IMG_SCALE = 0.82  

# --------------------------------------------------------------------------
# 1. COMPONENT DEFINITIONS
# --------------------------------------------------------------------------
UNIT_SCALE = {
    "pF": 1e-12, "nF": 1e-9, "uF": 1e-6, "µF": 1e-6, "mF": 1e-3, "F": 1.0,
    "nH": 1e-9, "uH": 1e-6, "µH": 1e-6, "mH": 1e-3, "H": 1.0,
    "ohm": 1.0, "Ω": 1.0, "kohm": 1e3, "kΩ": 1e3, "Mohm": 1e6, "MΩ": 1e6,
}

def mk(ctype, value, unit, box):
    return {"ctype": ctype, "value": value, "unit": unit, "box": box}

COMPONENTS = {
    "C3":  mk("C", 0.1,   "µF",  (72, 270, 107, 283)),
    "CMC": mk("L", 500,   "µH",  (206, 312, 241, 324)),
    "C2":  mk("C", 2.2,   "nF",  (303, 112, 337, 124)),
    "C4":  mk("C", 1,     "µF",  (303, 269, 337, 281)),
    "R4":  mk("R", 2.2,   "Ω",  (303, 429, 337, 441)),
    "C12": mk("C", 2.2,   "nF",  (303, 485, 337, 497)),
    "C9":  mk("C", 1,     "µF",  (381, 325, 416, 338)),
    "R2":  mk("R", 2.2,   "Ω",  (387, 237, 422, 249)),
    "C1":  mk("C", 2.2,   "nF",  (415, 78, 450, 90)),
    "R1":  mk("R", 2.2,   "Ω",  (415, 134, 450, 147)),
    "C11": mk("C", 2.2,   "nF",  (415, 451, 450, 463)),
    "C5":  mk("C", 33,    "µF",  (460, 269, 495, 281)),
    "DM":  mk("L", 10,    "µH",  (584, 168, 618, 180)),
    "C10": mk("C", 1,     "µF",  (606, 325, 641, 338)),
    "R3":  mk("R", 2.2,   "Ω",  (612, 237, 647, 249)),
    "C6":  mk("C", 10,    "µF",  (696, 280, 731, 293)),
    "C7":  mk("C", 1,     "µF",  (764, 280, 798, 293)),
    "C8":  mk("C", 0.1,   "µF",  (831, 280, 866, 293)),
}

DEFAULTS = {k: {"value": v["value"], "unit": v["unit"]} for k, v in COMPONENTS.items()}

ZS_DM = 100.0   
ZL_DM = 100.0
ZS_CM = 25.0    
ZL_CM = 25.0

FREQ_START = 1.0e4     
FREQ_STOP = 1.0e7      
N_POINTS = 800

CE102_LIMIT_FREQ = np.array([1.0e4, 5.0e4, 2.0e6, 1.0e7])
CE102_LIMIT_DBUV = np.array([94.0, 60.0, 60.0, 60.0])

NOISE_SRC_AT_10K = 110.0   
NOISE_SRC_AT_10M = 40.0    

NUM_DEVICES = 3
NUM_HARMONICS = 4
DEVICE_COLORS = ["orange", "teal", "magenta"]
DEVICE_LABELS = ["Dev 1", "Dev 2", "Dev 3"]

DEVICE_INIT_FREQS = [100e3, 500e3, 1e6]   
DEVICE_DEFAULT_AMP = 60.0

def noise_source_dbuv(freqs, lvl_10k, lvl_10m):
    logf = np.log10(freqs)
    lf0, lf1 = np.log10(FREQ_START), np.log10(FREQ_STOP)
    t = (logf - lf0) / (lf1 - lf0)
    return lvl_10k + t * (lvl_10m - lvl_10k)

def ce102_limit_curve(freqs):
    logf = np.log10(freqs)
    logbp = np.log10(CE102_LIMIT_FREQ)
    return np.interp(logf, logbp, CE102_LIMIT_DBUV)


# --------------------------------------------------------------------------
# 2. CIRCUIT MODEL  
# --------------------------------------------------------------------------
def z_cap(C, w):
    C = max(C, 1e-15)
    return 1.0 / (1j * w * C)

def z_ind(L, w):
    return 1j * w * L

def series_ABCD(Z):
    Z = np.asarray(Z, dtype=complex)
    N = Z.size
    out = np.tile(np.eye(2, dtype=complex), (N, 1, 1))
    out[:, 0, 1] = Z
    return out

def shunt_ABCD(Z):
    Z = np.asarray(Z, dtype=complex)
    N = Z.size
    out = np.tile(np.eye(2, dtype=complex), (N, 1, 1))
    Y = np.where(np.abs(Z) > 1e-30, 1.0 / Z, 1e30 + 0j)
    out[:, 1, 0] = Y
    return out

def cascade(mats):
    out = mats[0]
    for m in mats[1:]:
        out = out @ m
    return out

def insertion_loss_db(ABCD_stack, Zs, Zl):
    A = ABCD_stack[:, 0, 0]
    B = ABCD_stack[:, 0, 1]
    C = ABCD_stack[:, 1, 0]
    D = ABCD_stack[:, 1, 1]
    num = np.abs(A * Zl + B + C * Zs * Zl + D * Zs)
    den = np.abs(Zs + Zl)
    ratio = np.maximum(num / den, 1e-12)
    return 20.0 * np.log10(ratio)

def get_val(name):
    c = COMPONENTS[name]
    return c["value"] * UNIT_SCALE[c["unit"]]

def build_dm_chain(w):
    L_leak = 0.02 * get_val("CMC")     
    L_dm = get_val("DM")

    mats = [
        shunt_ABCD(z_cap(get_val("C3"), w)),
        series_ABCD(z_ind(L_leak, w)),
        shunt_ABCD(z_cap(get_val("C4"), w)),
        shunt_ABCD(get_val("R2") + z_cap(get_val("C9"), w)),
        shunt_ABCD(z_cap(get_val("C5"), w)),
        series_ABCD(z_ind(L_dm, w)),
        shunt_ABCD(get_val("R3") + z_cap(get_val("C10"), w)),
        shunt_ABCD(z_cap(get_val("C6") + get_val("C7") + get_val("C8"), w)),
    ]
    return cascade(mats)

def build_cm_chain(w):
    L_cm = get_val("CMC")
    L_dm = get_val("DM")

    Y_top = 1.0 / (get_val("R1") + z_cap(get_val("C1"), w)) + 1.0 / z_cap(get_val("C2"), w)
    Y_bot = 1.0 / (get_val("R4") + z_cap(get_val("C11"), w)) + 1.0 / z_cap(get_val("C12"), w)
    Z_y = 1.0 / (Y_top + Y_bot)

    mats = [
        series_ABCD(z_ind(2.0 * L_cm, w)),
        shunt_ABCD(Z_y),
        series_ABCD(z_ind(L_dm, w)),   
    ]
    return cascade(mats)

def compute_response(noise_10k=NOISE_SRC_AT_10K, noise_10m=NOISE_SRC_AT_10M):
    freqs = np.logspace(np.log10(FREQ_START), np.log10(FREQ_STOP), N_POINTS)
    ws = 2.0 * np.pi * freqs

    dm_stack = build_dm_chain(ws)
    cm_stack = build_cm_chain(ws)

    il_dm = insertion_loss_db(dm_stack, ZS_DM, ZL_DM)
    il_cm = insertion_loss_db(cm_stack, ZS_CM, ZL_CM)

    t_dm = 10.0 ** (-il_dm / 20.0)
    t_cm = 10.0 ** (-il_cm / 20.0)
    il_combined = -20.0 * np.log10(np.maximum(t_dm + t_cm, 1e-15))

    src = noise_source_dbuv(freqs, noise_10k, noise_10m)
    level_dm = src - il_dm
    level_cm = src - il_cm
    level_combined = src - il_combined
    limit = ce102_limit_curve(freqs)
    return freqs, level_dm, level_cm, level_combined, limit


def insertion_loss_at(freqs_at):
    """Combined (DM+CM) insertion loss in dB at arbitrary frequency points."""
    freqs_at = np.atleast_1d(np.asarray(freqs_at, dtype=float))
    ws = 2.0 * np.pi * freqs_at
    dm_stack = build_dm_chain(ws)
    cm_stack = build_cm_chain(ws)
    il_dm = insertion_loss_db(dm_stack, ZS_DM, ZL_DM)
    il_cm = insertion_loss_db(cm_stack, ZS_CM, ZL_CM)
    t_dm = 10.0 ** (-il_dm / 20.0)
    t_cm = 10.0 ** (-il_cm / 20.0)
    return -20.0 * np.log10(np.maximum(t_dm + t_cm, 1e-15))


# --------------------------------------------------------------------------
# 3. X-AXIS TICKS 
# --------------------------------------------------------------------------
XTICK_VALUES = [1e4, 2e4, 3e4, 5e4, 1e5, 2e5, 3e5, 5e5, 1e6, 2e6, 3e6, 5e6, 1e7]
XTICK_LABELS = ["10k", "20", "30", "50", "100k", "200", "300", "500",
                "1M", "2M", "3M", "5M", "10M"]


# --------------------------------------------------------------------------
# 4. TKINTER APPLICATION
# --------------------------------------------------------------------------
class CE102App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MIL-STD-461F CE102 EMI Filter Designer")

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        win_w = min(1650, sw - 40)
        win_h = min(950, sh - 100)
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(min(1350, win_w), min(850, win_h))

        self.entries = {}
        self.schem_image_ref = None
        self.pil_img = None
        self.noise_10k = NOISE_SRC_AT_10K
        self.noise_10m = NOISE_SRC_AT_10M
        self.device_freqs = list(DEVICE_INIT_FREQS)
        self.device_freq_entries = []
        self.device_enabled = [True] * NUM_DEVICES
        self.device_check_vars = []
        self.device_amplitudes = [DEVICE_DEFAULT_AMP] * NUM_DEVICES
        self.device_amp_entries = []
        self.device_amp_labels = []

        self.load_values()

        self._build_layout()
        self._draw_schematic()
        self._build_plot()
        self.recalculate()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ---------------------------------------------------------------
    def load_values(self):
        val_file = os.path.join(SCRIPT_DIR, "Values.txt")
        if not os.path.exists(val_file):
            return
        with open(val_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                name, val_str = line.split("=", 1)
                name = name.strip()
                val_str = val_str.strip()

                if name.startswith("DEV") and "_" in name:
                    devpart, field = name.split("_", 1)
                    idx_str = devpart[3:]
                    if not idx_str.isdigit():
                        continue
                    i = int(idx_str)
                    if not (0 <= i < NUM_DEVICES):
                        continue
                    try:
                        if field == "FQ":
                            self.device_freqs[i] = float(val_str) * 1e3
                        elif field == "DB":
                            self.device_amplitudes[i] = float(val_str)
                        elif field == "ON":
                            self.device_enabled[i] = int(val_str) == 1
                    except ValueError:
                        pass
                    continue

                if name in COMPONENTS:
                    comp = COMPONENTS[name]
                    base_units = {'C': 'F', 'L': 'H', 'R': 'ohm'}
                    base = base_units.get(comp['ctype'], '')
                    if val_str:
                        last = val_str[-1]
                        ends_with_unit = any(val_str.endswith(u) for u in UNIT_SCALE)
                        if last in "pnumkM" and not ends_with_unit:
                            val_str += base
                        elif last.isdigit() and comp['ctype'] == 'R' and not ends_with_unit:
                            val_str += "ohm"

                    parsed = self._parse_number_unit(val_str, comp["unit"], list(UNIT_SCALE.keys()))
                    if parsed is not None:
                        val, unit = parsed
                        comp["value"] = val
                        comp["unit"] = unit

    def save_values(self):
        val_file = os.path.join(SCRIPT_DIR, "Values.txt")
        with open(val_file, "w", encoding="utf-8") as f:
            for name, comp in COMPONENTS.items():
                val = comp["value"]
                if val == int(val):
                    val = int(val)
                f.write(f"{name}={val}{comp['unit']}\n")
            for i in range(NUM_DEVICES):
                freq_khz = (self.device_freqs[i] or 0) / 1e3
                if freq_khz == int(freq_khz):
                    freq_khz = int(freq_khz)
                amp = self.device_amplitudes[i]
                if amp == int(amp):
                    amp = int(amp)
                f.write(f"DEV{i}_FQ={freq_khz}\n")
                f.write(f"DEV{i}_DB={amp}\n")
                f.write(f"DEV{i}_ON={1 if self.device_enabled[i] else 0}\n")

    def on_closing(self):
        self.save_values()
        self.destroy()

    def save_images(self):
        graph_path = os.path.join(SCRIPT_DIR, "graph.png")
        self.fig.savefig(graph_path)

        if self.pil_img_orig:
            img_copy = self.pil_img_orig.copy()
            draw = ImageDraw.Draw(img_copy)
            try:
                font = ImageFont.truetype("arial.ttf", 11)
            except IOError:
                font = ImageFont.load_default()

            arr = np.array(self.pil_img_orig).astype(np.int32)
            dark_mask = arr.mean(axis=2) < 200

            def collisions(mask, y, x):
                h, w = mask.shape
                if x < 0 or y < 0 or x + w > dark_mask.shape[1] or y + h > dark_mask.shape[0]:
                    return None
                return int(dark_mask[y:y + h, x:x + w][mask].sum())

            for name, comp in COMPONENTS.items():
                x0, y0, x1, y1 = comp["box"]
                val_text = self._fmt_value(comp)

                try:
                    bbox = font.getbbox(val_text)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except AttributeError:
                    try:
                        tw, th = draw.textsize(val_text, font=font)
                    except AttributeError:
                        tw, th = 6 * len(val_text), 10
                tw += 2

                mask_im = Image.new("L", (tw + 8, th + 8), 255)
                mdraw = ImageDraw.Draw(mask_im)
                try:
                    mdraw.text((4, 4), val_text, font=font, fill=0)
                    scr = np.array(mask_im)
                    mask = scr[4:4 + th + 6, 4:4 + tw] < 200
                except Exception:
                    mask = None

                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                cands = [
                    ("center", (x0 + x1 - tw) // 2, (y0 + y1 - th) // 2),
                    ("above", cx - tw // 2, y0 - 3 - th),
                    ("below", cx - tw // 2, y1 + 3),
                    ("left", x0 - tw - 3, cy - th // 2),
                    ("right", x1 + 3, cy - th // 2),
                ]
                tx, ty = cands[0][1], cands[0][2]
                if mask is not None:
                    scored = []
                    for key, x, y in cands:
                        c = collisions(mask, y, x)
                        if c is not None:
                            scored.append((c, key, x, y))
                    if scored:
                        scored.sort(key=lambda s: (s[0], 0 if s[1] == "center" else 1))
                        _, _, tx, ty = scored[0]

                draw.text((tx, ty), val_text, fill=(0, 0, 200), font=font)
            
            circuit_path = os.path.join(SCRIPT_DIR, "circuit.png")
            img_copy.save(circuit_path)
            
        messagebox.showinfo("Saved", "circuit.png and graph.png saved successfully.")

    # ---------------------------------------------------------------
    def _build_layout(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # -------------------------------------------------------
        # LEFT COLUMN (Schematic Top, Params Bottom)
        # -------------------------------------------------------
        schem_w = int(Image.open(SCHEMATIC_PATH).size[0] * IMG_SCALE)
        left_col = ttk.Frame(outer, width=schem_w)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))
        left_col.pack_propagate(False)

        # 1. Schematic Panel (card)
        schem_card = ttk.LabelFrame(left_col, text=" Schematic ", padding=(8, 6))
        schem_card.pack(anchor="w", fill="x", pady=(0, 12))

        self.schem_canvas = tk.Canvas(schem_card, bg="white", highlightthickness=1,
                                      highlightbackground="#cccccc")
        self.schem_canvas.pack(anchor="nw")

        # 2+3. Component Parameters & Simulation Parameters (side by side)
        lower_row = ttk.Frame(left_col)
        lower_row.pack(anchor="w", fill="x")

        # 2. Component Parameter Tables (card)
        params_card = ttk.LabelFrame(lower_row, text=" Component Parameters ", padding=(8, 6))
        params_card.pack(side=tk.LEFT, anchor="n", fill="y")

        params_panel = ttk.Frame(params_card)
        params_panel.pack(anchor="nw")

        col_configs = [
            ("Capacitors", [f"C{i}" for i in range(1, 13)]),
            ("Inductors", ["CMC", "DM"]),
            ("Resistors", [f"R{i}" for i in range(1, 5)])
        ]

        for col_idx, (title, keys) in enumerate(col_configs):
            col_frame = ttk.Frame(params_panel)
            col_frame.grid(row=0, column=col_idx, sticky="n", padx=(0, 15))

            ttk.Label(col_frame, text=title, font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

            for r_idx, name in enumerate(keys, start=1):
                ttk.Label(col_frame, text=name, width=6, anchor="w").grid(row=r_idx, column=0, sticky="w", pady=1)
                ent = ttk.Entry(col_frame, width=8, justify="center")
                comp = COMPONENTS[name]
                ent.insert(0, self._fmt_value(comp))
                ent.grid(row=r_idx, column=1, sticky="w", pady=1)
                ent.bind("<Return>", lambda e, n=name: self._on_edit(n))
                ent.bind("<FocusOut>", lambda e, n=name: self._on_edit(n))
                self.entries[name] = ent

        # vertical separator between the two cards
        ttk.Separator(lower_row, orient="vertical").pack(side=tk.LEFT, fill="y", padx=14)

        # 3. Card panel: noise reference, device frequencies & save button
        card = ttk.LabelFrame(lower_row, text=" Simulation Parameters ",
                              padding=(10, 8))
        card.pack(side=tk.LEFT, anchor="n")

        inner = ttk.Frame(card)
        inner.pack(fill="x")

        # top row: noise reference (left) + save button (right)
        top_row = ttk.Frame(inner)
        top_row.pack(fill="x", pady=(0, 8))

        noise_frame = ttk.Frame(top_row)
        noise_frame.pack(side=tk.LEFT, anchor="n", padx=(0, 12))

        ttk.Label(noise_frame, text="Assumed EUT noise reference (dBuV):", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        ttk.Label(noise_frame, text="at 10 kHz:").grid(row=1, column=0, sticky="e", padx=(0, 4))
        self.noise_10k_entry = ttk.Entry(noise_frame, width=8)
        self.noise_10k_entry.insert(0, str(self.noise_10k))
        self.noise_10k_entry.grid(row=1, column=1, sticky="w")
        self.noise_10k_entry.bind("<Return>", self._on_noise_edit)
        self.noise_10k_entry.bind("<FocusOut>", self._on_noise_edit)

        ttk.Label(noise_frame, text="at 10 MHz:").grid(row=2, column=0, sticky="e", padx=(0, 4), pady=(4,0))
        self.noise_10m_entry = ttk.Entry(noise_frame, width=8)
        self.noise_10m_entry.insert(0, str(self.noise_10m))
        self.noise_10m_entry.grid(row=2, column=1, sticky="w", pady=(4,0))
        self.noise_10m_entry.bind("<Return>", self._on_noise_edit)
        self.noise_10m_entry.bind("<FocusOut>", self._on_noise_edit)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=(0, 8))

        # device frequency panel (below noise)
        dev_frame = ttk.Frame(inner)
        dev_frame.pack(anchor="w")

        ttk.Label(dev_frame, text="Connected device frequencies:",
                  font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, columnspan=6, sticky="w", pady=(0, 4))

        for i in range(NUM_DEVICES):
            amp_entry = ttk.Entry(dev_frame, width=6, justify="center")
            amp_entry.insert(0, f"{self.device_amplitudes[i]:g}")
            amp_entry.grid(row=i + 1, column=0, sticky="w", padx=(0, 4))
            amp_entry.bind("<Return>", lambda e, idx=i: self._on_device_amp_edit(idx))
            amp_entry.bind("<FocusOut>", lambda e, idx=i: self._on_device_amp_edit(idx))
            self.device_amp_entries.append(amp_entry)
            amp_label = ttk.Label(dev_frame, text="dBuV", anchor="e")
            amp_label.grid(row=i + 1, column=1, sticky="w", padx=(0, 8))
            self.device_amp_labels.append(amp_label)
            lbl = ttk.Label(dev_frame, text=f"{DEVICE_LABELS[i]}:", anchor="e")
            lbl.grid(row=i + 1, column=2, sticky="e", padx=(0, 4))
            freq_entry = ttk.Entry(dev_frame, width=8, justify="center")
            freq_entry.insert(0, f"{DEVICE_INIT_FREQS[i] / 1e3:g}")
            freq_entry.grid(row=i + 1, column=3, sticky="w", padx=(0, 3))
            freq_entry.bind("<Return>", lambda e, idx=i: self._on_device_freq_edit(idx))
            freq_entry.bind("<FocusOut>", lambda e, idx=i: self._on_device_freq_edit(idx))
            self.device_freq_entries.append(freq_entry)
            unit_label = ttk.Label(dev_frame, text="kHz", anchor="e")
            unit_label.grid(row=i + 1, column=4, sticky="w", padx=(0, 8))
            check_var = tk.BooleanVar(value=self.device_enabled[i])
            check = ttk.Checkbutton(dev_frame, text="Show", variable=check_var,
                                    command=lambda idx=i: self._on_device_check(idx))
            check.grid(row=i + 1, column=5, sticky="w")
            self.device_check_vars.append(check_var)

        ttk.Separator(left_col, orient='horizontal').pack(fill='x', pady=12)

        # Bottom button row of the left column
        button_row = ttk.Frame(left_col)
        button_row.pack(anchor="e")

        self.reset_button = ttk.Button(button_row, text="Reset Defaults",
                                       command=self.reset_defaults, width=14)
        self.reset_button.pack(side=tk.LEFT, padx=(0, 8))

        # Save button at the very bottom of the left column
        self.save_button = ttk.Button(button_row, text="Save Results",
                                      command=self.save_images, width=14)
        self.save_button.pack(side=tk.LEFT)

        # -------------------------------------------------------
        # RIGHT COLUMN (Graph spanning full height) - card
        # -------------------------------------------------------
        right_col = ttk.Frame(outer)
        # Expand=True ve fill=BOTH ile tüm yüksekliği doldurur
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        plot_card = ttk.LabelFrame(right_col, text=" Filter Response (CE102, 10 kHz - 10 MHz) ",
                                   padding=(8, 6))
        plot_card.pack(fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.Frame(plot_card)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)


    # ---------------------------------------------------------------
    def _draw_schematic(self):
        img = Image.open(SCHEMATIC_PATH)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)

        self.pil_img_orig = img.convert("RGB")

        arr = np.array(self.pil_img_orig).astype(np.int32)
        dist = np.abs(arr - np.array([255, 252, 248]))
        bg_mask = (dist.sum(axis=2) <= 24)
        arr[bg_mask] = [255, 255, 255]
        self.pil_img_orig = Image.fromarray(arr.astype(np.uint8), "RGB")

        w_orig, h_orig = self.pil_img_orig.size

        new_w = int(w_orig * IMG_SCALE)
        new_h = int(h_orig * IMG_SCALE)

        self.pil_img = self.pil_img_orig.resize((new_w, new_h), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        self.schem_image_ref = ImageTk.PhotoImage(self.pil_img)

        self.schem_canvas.delete("all")
        self.schem_canvas.config(width=new_w, height=new_h)
        self.schem_canvas.create_image(0, 0, anchor="nw", image=self.schem_image_ref)

    @staticmethod
    def _fmt_value(comp):
        v = comp["value"]
        if v == int(v):
            v = int(v)
        return f"{v}{comp['unit']}"

    # ---------------------------------------------------------------
    def _parse_number_unit(self, text, default_unit, valid_units):
        text = text.strip().replace(" ", "").replace("\u00b5", "u")
        if text == "":
            return None
        i = len(text)
        while i > 0 and not (text[i - 1].isdigit() or text[i - 1] == '.'):
            i -= 1
        num_part, unit_part = text[:i], text[i:]
        try:
            val = float(num_part)
        except ValueError:
            return None
        if val < 0:
            return None
        if unit_part == "":
            return val, default_unit
        norm = {
            "pf": "pF", "pF": "pF", "PF": "pF",
            "nf": "nF", "nF": "nF", "NF": "nF",
            "uf": "µF", "uF": "µF", "UF": "µF",
            "mf": "mF", "mF": "mF", "MF": "mF",
            "nh": "nH", "nH": "nH", "NH": "nH",
            "uh": "µH", "uH": "µH", "UH": "µH",
            "mh": "mH", "mH": "mH", "MH": "mH",
            "ohm": "Ω", "Ohm": "Ω", "OHM": "Ω", "Ω": "Ω", "R": "Ω", "r": "Ω",
            "kohm": "kΩ", "Kohm": "kΩ", "KOHM": "kΩ", "kΩ": "kΩ", "KΩ": "kΩ",
            "k": "kΩ", "K": "kΩ",
            "Mohm": "MΩ", "MOHM": "MΩ", "MΩ": "MΩ",
        }
        unit = norm.get(unit_part, unit_part)
        if unit not in valid_units:
            unit = default_unit
        return val, unit

    def _on_edit(self, name):
        ent = self.entries[name]
        text = ent.get()
        comp = COMPONENTS[name]
        parsed = self._parse_number_unit(text, comp["unit"], UNIT_SCALE.keys())
        if parsed is None:
            messagebox.showwarning("Invalid value",
                                    f"Could not parse value for {name}: '{text}'.\nKeeping previous value.")
            ent.delete(0, tk.END)
            ent.insert(0, self._fmt_value(comp))
            return
        val, unit = parsed
        comp["value"] = val
        comp["unit"] = unit
        ent.delete(0, tk.END)
        ent.insert(0, self._fmt_value(comp))
        self.recalculate()

    def _on_noise_edit(self, event=None):
        try:
            v10k = float(self.noise_10k_entry.get().strip())
            v10m = float(self.noise_10m_entry.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid value", "Noise reference levels must be numeric (dBuV).")
            self.noise_10k_entry.delete(0, tk.END)
            self.noise_10k_entry.insert(0, str(self.noise_10k))
            self.noise_10m_entry.delete(0, tk.END)
            self.noise_10m_entry.insert(0, str(self.noise_10m))
            return
        self.noise_10k, self.noise_10m = v10k, v10m
        self.recalculate()

    def _on_device_freq_edit(self, index, event=None):
        ent = self.device_freq_entries[index]
        text = ent.get().strip()
        try:
            khz = float(text)
        except ValueError:
            ent.delete(0, tk.END)
            ent.insert(0, f"{(self.device_freqs[index] or 0) / 1e3:g}")
            return
        khz = min(max(khz, 1.0), 1e5)
        self.device_freqs[index] = khz * 1e3
        ent.delete(0, tk.END)
        ent.insert(0, f"{khz:g}")
        self.recalculate()

    def _on_device_amp_edit(self, index, event=None):
        ent = self.device_amp_entries[index]
        text = ent.get().strip()
        if text.lower().endswith("dbuv"):
            text = text[:-5]
        try:
            val = float(text)
        except ValueError:
            ent.delete(0, tk.END)
            ent.insert(0, str(self.device_amplitudes[index]))
            return
        val = min(max(val, 0.0), 200.0)
        self.device_amplitudes[index] = val
        ent.delete(0, tk.END)
        ent.insert(0, f"{val:g}")
        self.recalculate()

    def _on_device_check(self, index):
        self.device_enabled[index] = self.device_check_vars[index].get()
        self.recalculate()

    @staticmethod
    def _harmonic_frequencies(freq, num=NUM_HARMONICS):
        if freq is None or freq <= 0:
            return []
        return [freq * n for n in range(1, num + 1)]

    def reset_defaults(self):
        for name, comp in DEFAULTS.items():
            COMPONENTS[name]["value"] = comp["value"]
            COMPONENTS[name]["unit"] = comp["unit"]
            ent = self.entries[name]
            ent.delete(0, tk.END)
            ent.insert(0, self._fmt_value(COMPONENTS[name]))
        self.noise_10k, self.noise_10m = NOISE_SRC_AT_10K, NOISE_SRC_AT_10M
        self.noise_10k_entry.delete(0, tk.END)
        self.noise_10k_entry.insert(0, str(self.noise_10k))
        self.noise_10m_entry.delete(0, tk.END)
        self.noise_10m_entry.insert(0, str(self.noise_10m))
        self.device_freqs = list(DEVICE_INIT_FREQS)
        self.device_amplitudes = [DEVICE_DEFAULT_AMP] * NUM_DEVICES
        self.device_enabled = [True] * NUM_DEVICES
        for i in range(NUM_DEVICES):
            ent = self.device_freq_entries[i]
            ent.delete(0, tk.END)
            ent.insert(0, f"{self.device_freqs[i] / 1e3:g}")
            ent = self.device_amp_entries[i]
            ent.delete(0, tk.END)
            ent.insert(0, f"{self.device_amplitudes[i]:g}")
            self.device_check_vars[i].set(True)
        self.recalculate()

    # ---------------------------------------------------------------
    def _build_plot(self):
        # Increased figsize height slightly since it spans full right side
        self.fig = Figure(figsize=(10.5, 7.5), dpi=100) 
        self.ax = self.fig.add_subplot(111)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_plot, self.plot_frame)
        toolbar.update()

    def recalculate(self):
        try:
            freqs, level_dm, level_cm, level_combined, limit = compute_response(self.noise_10k, self.noise_10m)
        except Exception as ex:
            messagebox.showerror("Calculation error", str(ex))
            return

        ax = self.ax
        ax.clear()

        # Combined drawn first, CM drawn LAST (on top) so the green curve
        # stays visible even where it nearly coincides with the combined level.
        ax.semilogx(freqs, level_dm, color="tab:blue", linewidth=1.4, label="Differential Mode")
        ax.semilogx(freqs, level_combined, color="tab:purple", linewidth=1.7, label="DM + CM Combined")
        ax.semilogx(freqs, level_cm, color="tab:green", linestyle="-", linewidth=1.5, label="Common Mode")
        ax.semilogx(freqs, limit, color="red", linestyle="--", linewidth=1.6)
        ax.text(freqs[-1], limit[-1] + 2, "CE102-28V", color="red", fontsize=9,
                ha="right", va="bottom")

        ax.set_xlim(FREQ_START, FREQ_STOP)
        ax.xaxis.set_major_locator(FixedLocator(XTICK_VALUES))
        ax.xaxis.set_major_formatter(FixedFormatter(XTICK_LABELS))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Frequency in Hz")
        ax.set_ylabel("Level in dBuV")

        ymin, ymax = -20, 100

        # Device fundamental + harmonic markers, with levels AFTER filtering
        device_levels = []
        for i in range(NUM_DEVICES):
            if self.device_freqs[i] is None or not self.device_enabled[i]:
                continue
            harmonics = [h for h in self._harmonic_frequencies(self.device_freqs[i])
                         if FREQ_START <= h <= FREQ_STOP]
            if not harmonics:
                continue
            amp = self.device_amplitudes[i]
            il = insertion_loss_at(np.array(harmonics))
            device_levels.append(amp - il)

        if device_levels:
            flat = [lv for grp in device_levels for lv in grp]
            if flat:
                ymax = max(ymax, max(flat) + 10)
                if min(flat) < ymin:
                    ymin = min(flat) - 10

        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(FixedLocator(np.arange(ymin, ymax + 1, 10)))

        for i in range(NUM_DEVICES):
            if self.device_freqs[i] is None or not self.device_enabled[i]:
                continue
            color = DEVICE_COLORS[i]
            label = DEVICE_LABELS[i]
            amp = self.device_amplitudes[i]
            harmonics = [h for h in self._harmonic_frequencies(self.device_freqs[i])
                         if FREQ_START <= h <= FREQ_STOP]
            if not harmonics:
                continue
            hfs = np.array(harmonics)
            il = insertion_loss_at(hfs)
            lev = amp - il
            if self.device_amp_labels:
                self.device_amp_labels[i].config(
                    text=f"dBuV\n(-> {max(lev):.0f} dBuV)")

            # Small per-device x jitter separates coincident harmonics
            jitter = 1.0 + (i - 1) * 0.02
            for hi, (hf, lv) in enumerate(zip(hfs, lev)):
                fx = hf * jitter
                if hi == 0:
                    ax.axvline(fx, color=color, linewidth=1.8, alpha=0.9)
                    ax.text(fx, ymax - 4, f"{label} f1", color=color, fontsize=8,
                            ha="center", va="top", rotation=90)
                else:
                    ax.axvline(fx, color=color, linewidth=1.4, alpha=0.75, linestyle=":")
                    ax.text(fx, ymin + 4, f"{hi + 1}f", color=color, fontsize=7,
                            ha="center", va="bottom", rotation=90)
                ax.plot(fx, lv, marker="o", markersize=5, color=color,
                        linestyle="none", clip_on=False)
                # short horizontal tick at the filtered level (no full-width line)
                ax.plot([fx * 0.97, fx * 1.03], [lv, lv], color=color,
                        linewidth=1.0, alpha=0.8, clip_on=False)

        ax.grid(True, which="major", linestyle=":", linewidth=0.6)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right", fontsize=8, ncol=3)
        ax.set_title("Differential-Mode / Common-Mode / Combined Level vs. CE102-28V Limit", fontsize=10)

        self.fig.tight_layout()
        self.canvas_plot.draw()


def main():
    if not os.path.isfile(SCHEMATIC_PATH):
        raise SystemExit(f"Cannot find {SCHEMATIC_PATH}. Make sure SCH.png is in the same folder as this script.")
    app = CE102App()
    app.mainloop()


if __name__ == "__main__":
    main()
