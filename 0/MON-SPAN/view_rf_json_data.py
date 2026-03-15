import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Update this if your files are in a different folder:
DATA_GLOB = "2023-09-22 00-00-0*.json"  # matches 00..05

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def freq_axis(center_hz, span_hz, n_bins):
    # Bins are evenly spaced across [center - span/2, center + span/2]
    start = center_hz - span_hz / 2.0
    stop = center_hz + span_hz / 2.0
    return np.linspace(start, stop, n_bins)

def summarize_record(rec):
    # Pull block 1
    s1 = np.array(rec["spectrum_01"], dtype=float)
    c1 = float(rec["center_01"])
    sp1 = float(rec["span_01"])
    r1 = float(rec["res_01"])
    pga1 = rec.get("pga_01", None)

    # Pull block 2
    s2 = np.array(rec["spectrum_02"], dtype=float)
    c2 = float(rec["center_02"])
    sp2 = float(rec["span_02"])
    r2 = float(rec["res_02"])
    pga2 = rec.get("pga_02", None)

    return {
        "start_time": rec.get("start_time", ""),
        "numRfBlocks": rec.get("numRfBlocks", ""),
        "block1": {
            "center_hz": c1, "span_hz": sp1, "res_hz": r1, "pga": pga1,
            "n_bins": len(s1),
            "min": float(np.min(s1)), "max": float(np.max(s1)),
            "mean": float(np.mean(s1)), "std": float(np.std(s1)),
            "peak_bin": int(np.argmax(s1))
        },
        "block2": {
            "center_hz": c2, "span_hz": sp2, "res_hz": r2, "pga": pga2,
            "n_bins": len(s2),
            "min": float(np.min(s2)), "max": float(np.max(s2)),
            "mean": float(np.mean(s2)), "std": float(np.std(s2)),
            "peak_bin": int(np.argmax(s2))
        }
    }

def print_summary(summaries):
    print("\n=== SUMMARY (one row per timestamp) ===\n")
    for s in summaries:
        b1 = s["block1"]
        b2 = s["block2"]
        print(
            f"{s['start_time']} | blocks={s['numRfBlocks']} | "
            f"B1 center={b1['center_hz']:.0f} span={b1['span_hz']:.0f} res={b1['res_hz']:.0f} "
            f"max={b1['max']:.1f} peak_bin={b1['peak_bin']} | "
            f"B2 center={b2['center_hz']:.0f} span={b2['span_hz']:.0f} res={b2['res_hz']:.0f} "
            f"max={b2['max']:.1f} peak_bin={b2['peak_bin']}"
        )

def plot_lines_over_time(records, summaries):
    # Plot each timestamp as its own line (Block 1 and Block 2)
    plt.figure()
    for rec, s in zip(records, summaries):
        y = np.array(rec["spectrum_01"], dtype=float)
        x = freq_axis(s["block1"]["center_hz"], s["block1"]["span_hz"], len(y))
        plt.plot(x/1e6, y, label=s["start_time"])
    plt.title("Block 1: Spectrum vs Frequency (all timestamps)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Amplitude (raw units)")
    plt.legend(fontsize=7)
    plt.tight_layout()

    plt.figure()
    for rec, s in zip(records, summaries):
        y = np.array(rec["spectrum_02"], dtype=float)
        x = freq_axis(s["block2"]["center_hz"], s["block2"]["span_hz"], len(y))
        plt.plot(x/1e6, y, label=s["start_time"])
    plt.title("Block 2: Spectrum vs Frequency (all timestamps)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Amplitude (raw units)")
    plt.legend(fontsize=7)
    plt.tight_layout()

def plot_waterfall(records, summaries, block_key="spectrum_01"):
    # Heatmap over time: rows = timestamps, cols = frequency bins
    mat = np.array([np.array(r[block_key], dtype=float) for r in records])
    times = [s["start_time"] for s in summaries]

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.title(f"Waterfall Heatmap: {block_key} (time x bins)")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Time (index)")
    plt.yticks(range(len(times)), times, fontsize=7)
    plt.colorbar(label="Amplitude (raw units)")
    plt.tight_layout()

def main():
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No files found matching: {DATA_GLOB}")
        print("Put this script in the same folder as the JSON files, or change DATA_GLOB.")
        return

    records = [load_json(p) for p in paths]
    summaries = [summarize_record(r) for r in records]

    print("Loaded files:")
    for p in paths:
        print(" -", os.path.basename(p))

    print_summary(summaries)

    plot_lines_over_time(records, summaries)
    plot_waterfall(records, summaries, "spectrum_01")
    plot_waterfall(records, summaries, "spectrum_02")

    plt.show()

if __name__ == "__main__":
    main()