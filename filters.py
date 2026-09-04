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
    "pF": 1e-12, "nF": 1e-9, "uF": 1e-6, "mF": 1e-3, "F": 1.0,
    "nH": 1e-9, "uH": 1e-6, "mH": 1e-3, "H": 1.0,
    "ohm": 1.0, "kohm": 1e3, "Mohm": 1e6,
}

def mk(ctype, value, unit, box):
    return {"ctype": ctype, "value": value, "unit": unit, "box": box}

COMPONENTS = {
    "C3":  mk("C", 0.1,   "uF",  (72, 270, 107, 283)),
    "CMC": mk("L", 500,   "uH",  (206, 312, 241, 324)),
    "C2":  mk("C", 2.2,   "nF",  (303, 112, 337, 124)),
    "C4":  mk("C", 1,     "uF",  (303, 269, 337, 281)),
    "R4":  mk("R", 2.2,   "ohm", (303, 429, 337, 441)),
    "C12": mk("C", 2.2,   "nF",  (303, 485, 337, 497)),
    "C9":  mk("C", 1,     "uF",  (381, 325, 416, 338)),
    "R2":  mk("R", 2.2,   "ohm", (387, 237, 422, 249)),
    "C1":  mk("C", 2.2,   "nF",  (415, 78, 450, 90)),
    "R1":  mk("R", 2.2,   "ohm", (415, 134, 450, 147)),
    "C11": mk("C", 2.2,   "nF",  (415, 451, 450, 463)),
    "C5":  mk("C", 33,    "uF",  (460, 269, 495, 281)),
    "DM":  mk("L", 10,    "uH",  (584, 168, 618, 180)),
    "C10": mk("C", 1,     "uF",  (606, 325, 641, 338)),
    "R3":  mk("R", 2.2,   "ohm", (612, 237, 647, 249)),
    "C6":  mk("C", 10,    "uF",  (696, 280, 731, 293)),
    "C7":  mk("C", 1,     "uF",  (764, 280, 798, 293)),
    "C8":  mk("C", 0.1,   "uF",  (831, 280, 866, 293)),
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
    return np.array([[1.0, Z], [0.0, 1.0]], dtype=complex)

def shunt_ABCD(Z):
    Y = 1.0 / Z if abs(Z) > 1e-30 else 1e30
    return np.array([[1.0, 0.0], [Y, 1.0]], dtype=complex)

def cascade(mats):
    out = np.eye(2, dtype=complex)
    for m in mats:
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

    dm_stack = np.empty((N_POINTS, 2, 2), dtype=complex)
    cm_stack = np.empty((N_POINTS, 2, 2), dtype=complex)
    for i, w in enumerate(ws):
        dm_stack[i] = build_dm_chain(w)
        cm_stack[i] = build_cm_chain(w)

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
        self.geometry("1650x950")
        self.minsize(1350, 850)

        self.entries = {}
        self.schem_image_ref = None
        self.pil_img = None
        self.noise_10k = NOISE_SRC_AT_10K
        self.noise_10m = NOISE_SRC_AT_10M

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
        with open(val_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                name, val_str = line.split("=", 1)
                name = name.strip()
                val_str = val_str.strip()
                
                if name in COMPONENTS:
                    comp = COMPONENTS[name]
                    base_units = {'C': 'F', 'L': 'H', 'R': 'ohm'}
                    if val_str and val_str[-1] in "pnumkM":
                        val_str += base_units.get(comp['ctype'], '')
                    elif val_str and val_str[-1].isdigit() and comp['ctype'] == 'R':
                        val_str += "ohm"
                    
                    parsed = self._parse_number_unit(val_str, comp["unit"], list(UNIT_SCALE.keys()))
                    if parsed is not None:
                        val, unit = parsed
                        comp["value"] = val
                        comp["unit"] = unit

    def save_values(self):
        val_file = os.path.join(SCRIPT_DIR, "Values.txt")
        with open(val_file, "w") as f:
            for name, comp in COMPONENTS.items():
                unit_str = comp["unit"]
                unit_abbr = unit_str.replace("F", "").replace("H", "").replace("ohm", "")
                val = comp["value"]
                if val == int(val):
                    val = int(val)
                f.write(f"{name}={val}{unit_abbr}\n")

    def on_closing(self):
        self.save_values()
        self.destroy()

    def save_images(self):
        graph_path = os.path.join(SCRIPT_DIR, "graph.png")
        self.fig.savefig(graph_path)

        if self.pil_img:
            img_copy = self.pil_img.copy()
            draw = ImageDraw.Draw(img_copy)
            try:
                font = ImageFont.truetype("arial.ttf", 11) 
            except IOError:
                font = ImageFont.load_default()

            for name, comp in COMPONENTS.items():
                x0, y0, x1, y1 = [int(v * IMG_SCALE) for v in comp["box"]]
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
                
                tx = (x0 + x1) / 2 - tw / 2
                ty = (y0 + y1) / 2 - th / 2
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
        left_col = ttk.Frame(outer)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))

        # 1. Schematic Panel
        schem_hdr = ttk.Label(left_col, text="Schematic (SCH.png)", font=("Segoe UI", 11, "bold"))
        schem_hdr.pack(anchor="w", pady=(0, 4))

        self.schem_canvas = tk.Canvas(left_col, bg="white", highlightthickness=0)
        self.schem_canvas.pack(anchor="nw", pady=(0, 12))

        # 2. Component Parameter Tables
        params_panel = ttk.Frame(left_col)
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
                
        ttk.Separator(left_col, orient='horizontal').pack(fill='x', pady=15)

        # 3. Noise-source reference textboxes & Save button 
        noise_frame = ttk.Frame(left_col)
        noise_frame.pack(anchor="w", pady=(0, 0))
        
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

        ttk.Button(noise_frame, text="Save Results", command=self.save_images, width=12).grid(row=1, column=2, rowspan=2, padx=(20, 0), sticky="ns")

        # -------------------------------------------------------
        # RIGHT COLUMN (Graph spanning full height)
        # -------------------------------------------------------
        right_col = ttk.Frame(outer)
        # Expand=True ve fill=BOTH ile tüm yüksekliği doldurur
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        hdr2 = ttk.Label(right_col, text="Filter Response  (CE102, 10 kHz - 10 MHz)", font=("Segoe UI", 11, "bold"))
        hdr2.pack(anchor="w", pady=(0, 2))

        self.plot_frame = ttk.Frame(right_col)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)


    # ---------------------------------------------------------------
    def _draw_schematic(self):
        self.pil_img_orig = Image.open(SCHEMATIC_PATH).convert("RGB")
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
            "pf": "pF", "PF": "pF", "nf": "nF", "NF": "nF",
            "uf": "uF", "UF": "uF", "mf": "mF",
            "nh": "nH", "NH": "nH", "uh": "uH", "UH": "uH", "mh": "mH", "MH": "mH",
            "ohm": "ohm", "Ohm": "ohm", "OHM": "ohm", "R": "ohm", "r": "ohm",
            "kohm": "kohm", "Kohm": "kohm", "KOHM": "kohm", "k": "kohm", "K": "kohm",
            "Mohm": "Mohm", "MOHM": "Mohm",
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

        ax.semilogx(freqs, level_dm, color="tab:blue", linewidth=1.4, label="Differential Mode")
        ax.semilogx(freqs, level_cm, color="tab:green", linewidth=1.4, label="Common Mode")
        ax.semilogx(freqs, level_combined, color="tab:purple", linewidth=1.7, label="DM + CM Combined")
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
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(FixedLocator(np.arange(ymin, ymax + 1, 10)))

        ax.grid(True, which="major", linestyle=":", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
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