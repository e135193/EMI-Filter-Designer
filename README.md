# CE102 EMI Power-Line Filter Designer

Interactive **MIL-STD-461F/G CE102** filter design and analysis tool. It models a conducted-emissions power-line EMI filter and visualizes its performance against the CE102 (10 kHz – 10 MHz) emission limit, while letting you simulate the frequency and amplitude of connected devices and their harmonics.

---

## Features

- **CE102 limit compliance** – Plots differential-mode (DM), common-mode (CM), and combined (DM + CM) emission levels against the MIL-STD-461 CE102-28V limit curve.
- **Interactive schematic** – Displays the filter schematic (`SCH.png`) with live component value labels.
- **Component editing** – Capacitors (C1–C12), inductors (CMC, DM), and resistors (R1–R4) can be edited directly in the GUI; values persist to `Values.txt` on exit.
- **Noise source reference** – Configurable assumed EUT noise level at 10 kHz and 10 MHz.
- **Device frequency simulation** – Up to three connected devices, each with:
  - Amplitude in **dBµV**
  - Fundamental frequency in **kHz**
  - A **Show** checkbox to toggle visibility on the graph
- **Harmonic analysis** – The fundamental and its 2nd, 3rd, and 4th harmonics are drawn as vertical markers on the graph, with amplitude points at each harmonic.
- **Results export** – Saves the filter response graph (`graph.png`) and the annotated schematic (`circuit.png`).

---

## Circuit Model

The tool models a conventional EMI power-line filter using **chain (ABCD) matrix** analysis. It computes two independent paths and combines them:

- **Differential Mode (DM):** leakage inductance of the common-mode choke, X-caps, the DM choke, and associated resistors/capacitors.
- **Common Mode (CM):** common-mode choke inductance, Y-capacitors, and the DM choke.

From the cascade of two-port ABCD matrices, the **insertion loss** is calculated for the DM and CM paths, then the two are combined to estimate the total attenuation and residual emission level.

---

## Usage

### Requirements

- Python 3.8+
- `numpy`
- `Pillow` (PIL)
- `matplotlib`

Install dependencies:

```bash
pip install numpy pillow matplotlib
```

### Running

```bash
python filters.py
```

> **Note:** A `SCH.png` schematic image must be present in the same folder as the script.

### Using the GUI

1. **Component Parameters** – Adjust capacitor, inductor, and resistor values.
2. **Assumed EUT noise reference** – Set the noise source level at 10 kHz and 10 MHz.
3. **Connected device frequencies** – Enter each device's amplitude (dBµV) and frequency (kHz), and toggle its **Show** checkbox.
4. **Filter Response graph** – Watch DM, CM, and combined levels update in real time, with device fundamentals and harmonics overlaid.
5. **Save Results** – Export `graph.png` and `circuit.png`.

---

## Files

| File | Description |
|------|-------------|
| `filters.py` | Main application (single file, no external source files required) |
| `SCH.png` | Filter schematic image displayed in the UI |
| `Values.txt` | Auto-generated/saved component values |
| `graph.png` | Exported filter response plot |
| `circuit.png` | Exported annotated schematic |

---

## Disclaimer

This tool is intended for **engineering analysis and education**. It provides an idealized model and is **not a substitute for compliance testing** in an accredited laboratory. Always validate final designs against the official MIL-STD-461 test procedures.

![EMI Filter Designer](https://github.com/e135193/EMI-Filter-Designer-/blob/main/GUI.png)
