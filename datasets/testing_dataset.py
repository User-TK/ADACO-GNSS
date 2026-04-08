import zipfile
import json
import h5py
import numpy as np
from pathlib import Path
from io import TextIOWrapper

# ── CONFIG ───────────────────────────────────────────────────────────────
BASE_PATH = Path("Dataset")

DATASETS = [
    ("21",   "2023-09-21", 0),
    ("22",   "2023-09-22", 0),
    ("23",   "2023-09-23", 0),
    ("24",   "2023-09-24", 0),
    ("25",   "2023-09-25", 0),
    ("26",   "2023-09-26", 0),
    ("27",   "2023-09-27", 0),
    ("28",   "2023-09-28", 0),
    ("29",   "2023-09-29", 0),
    ("30",   "2023-09-30", 0),
    ("1221", "2023-12-21", 1),
]

OUTPUT_HDF5      = "gnss_dataset.h5"
MAX_SATELLITES   = 66
MAX_RAWX_MEAS    = 60
SECONDS_PER_HOUR = 3600
# ─────────────────────────────────────────────────────────────────────────

PVT_FIELDS     = ["fixType", "gnssFixOk", "numSV", "lat", "lon",
                  "height", "hMSL", "hAcc", "vAcc", "gSpeed",
                  "pDOP", "carrSoln", "difSoln", "invalidLlh", "tAcc"]
CLOCK_FIELDS   = ["clkB", "clkD", "tAcc", "fAcc"]
DOP_FIELDS     = ["gDOP", "pDOP", "tDOP", "vDOP", "hDOP", "nDOP", "eDOP"]
POSECEF_FIELDS = ["ecefX", "ecefY", "ecefZ", "pAcc"]
SPAN_META      = ["pga_01", "pga_02", "center_01", "center_02",
                  "span_01", "span_02", "res_01", "res_02"]
SAT_FIELDS     = ["gnssId", "svId", "cno", "elev", "azim",
                  "prRes", "qualityInd", "svUsed", "health", "ephAvail"]
RAWX_FIELDS    = ["prMes", "cpMes", "doMes", "gnssId", "svId",
                  "cno", "prStd", "cpStd", "doStd", "prValid", "cpValid"]


def make_filename(date_str, hour, sec):
    """Build filename like: 2023-09-21 00-00-05.json"""
    mm = sec // 60
    ss = sec % 60
    return f"{date_str} {hour:02d}-{mm:02d}-{ss:02d}.json"


def read_json_from_zip(zf, path):
    try:
        with zf.open(path) as f:
            raw = json.load(TextIOWrapper(f, encoding="utf-8"))
            return raw.get("root", raw)
    except Exception:
        return {}


def parse_mon_span(d):
    s1   = np.resize(np.array(d.get("spectrum_01", [0]*256), dtype=np.float32), 256)
    s2   = np.resize(np.array(d.get("spectrum_02", [0]*256), dtype=np.float32), 256)
    meta = np.array([d.get(f, 0) for f in SPAN_META], dtype=np.float32)
    return s1, s2, meta


def parse_scalar(d, fields):
    return np.array([d.get(f, 0) for f in fields], dtype=np.float32)


def parse_nav_sat(d):
    out = np.zeros((MAX_SATELLITES, len(SAT_FIELDS)), dtype=np.float32)
    i, row = 1, 0
    while f"svId_{i:02d}" in d and row < MAX_SATELLITES:
        if d.get(f"svId_{i:02d}", 0) != 255:
            for j, field in enumerate(SAT_FIELDS):
                out[row, j] = d.get(f"{field}_{i:02d}", 0)
            row += 1
        i += 1
    return out


def parse_rxm_rawx(d):
    out = np.zeros((MAX_RAWX_MEAS, len(RAWX_FIELDS)), dtype=np.float32)
    i, row = 1, 0
    while f"prMes_{i:02d}" in d and row < MAX_RAWX_MEAS:
        for j, field in enumerate(RAWX_FIELDS):
            out[row, j] = d.get(f"{field}_{i:02d}", 0)
        row += 1
        i += 1
    return out


def count_samples():
    total = 0
    for day, _, _ in DATASETS:
        for hour in range(24):
            if (BASE_PATH / day / f"{hour}.zip").exists():
                total += SECONDS_PER_HOUR
    return total


# ── MAIN ─────────────────────────────────────────────────────────────────
print("Counting samples...")
N = count_samples()
print(f"Total samples: {N:,}")

with h5py.File(OUTPUT_HDF5, "w") as hf:
    chunk = min(3600, N)

    ds_spec1 = hf.create_dataset("spectrum_01", shape=(N, 256),
                                  dtype=np.float32, chunks=(chunk, 256),
                                  compression="gzip", compression_opts=4)
    ds_spec2 = hf.create_dataset("spectrum_02", shape=(N, 256),
                                  dtype=np.float32, chunks=(chunk, 256),
                                  compression="gzip", compression_opts=4)
    ds_span  = hf.create_dataset("span_meta",   shape=(N, len(SPAN_META)),      dtype=np.float32)
    ds_clock = hf.create_dataset("nav_clock",   shape=(N, len(CLOCK_FIELDS)),   dtype=np.float32)
    ds_dop   = hf.create_dataset("nav_dop",     shape=(N, len(DOP_FIELDS)),     dtype=np.float32)
    ds_pos   = hf.create_dataset("nav_posecef", shape=(N, len(POSECEF_FIELDS)), dtype=np.float32)
    ds_pvt   = hf.create_dataset("nav_pvt",     shape=(N, len(PVT_FIELDS)),     dtype=np.float32)
    ds_sat   = hf.create_dataset("nav_sat",     shape=(N, MAX_SATELLITES, len(SAT_FIELDS)),
                                  dtype=np.float32,
                                  chunks=(chunk, MAX_SATELLITES, len(SAT_FIELDS)),
                                  compression="gzip", compression_opts=4)
    ds_rawx  = hf.create_dataset("rxm_rawx",    shape=(N, MAX_RAWX_MEAS, len(RAWX_FIELDS)),
                                  dtype=np.float32,
                                  chunks=(chunk, MAX_RAWX_MEAS, len(RAWX_FIELDS)),
                                  compression="gzip", compression_opts=4)
    ds_label = hf.create_dataset("label",       shape=(N,), dtype=np.int8)
    ds_hour  = hf.create_dataset("hour",        shape=(N,), dtype=np.int8)
    ds_day   = hf.create_dataset("day",         shape=(N,), dtype="S4")

    for ds, fields in [(ds_span,  SPAN_META),    (ds_clock, CLOCK_FIELDS),
                       (ds_dop,   DOP_FIELDS),   (ds_pos,   POSECEF_FIELDS),
                       (ds_pvt,   PVT_FIELDS),   (ds_sat,   SAT_FIELDS),
                       (ds_rawx,  RAWX_FIELDS)]:
        ds.attrs["fields"] = fields

    cursor = 0

    for day, date_str, label in DATASETS:
        day_path = BASE_PATH / day
        if not day_path.exists():
            print(f"Skipping missing: {day_path}")
            continue

        print(f"\nDay {day} (label={label})")

        for hour in range(24):
            zip_path = day_path / f"{hour}.zip"
            if not zip_path.exists():
                continue

            print(f"  Hour {hour}...", end="\r")

            b_spec1 = np.zeros((SECONDS_PER_HOUR, 256),                                dtype=np.float32)
            b_spec2 = np.zeros((SECONDS_PER_HOUR, 256),                                dtype=np.float32)
            b_span  = np.zeros((SECONDS_PER_HOUR, len(SPAN_META)),                     dtype=np.float32)
            b_clock = np.zeros((SECONDS_PER_HOUR, len(CLOCK_FIELDS)),                  dtype=np.float32)
            b_dop   = np.zeros((SECONDS_PER_HOUR, len(DOP_FIELDS)),                    dtype=np.float32)
            b_pos   = np.zeros((SECONDS_PER_HOUR, len(POSECEF_FIELDS)),                dtype=np.float32)
            b_pvt   = np.zeros((SECONDS_PER_HOUR, len(PVT_FIELDS)),                    dtype=np.float32)
            b_sat   = np.zeros((SECONDS_PER_HOUR, MAX_SATELLITES,  len(SAT_FIELDS)),   dtype=np.float32)
            b_rawx  = np.zeros((SECONDS_PER_HOUR, MAX_RAWX_MEAS,   len(RAWX_FIELDS)), dtype=np.float32)

            with zipfile.ZipFile(zip_path) as zf:
                all_files = set(zf.namelist())

                for sec in range(SECONDS_PER_HOUR):
                    fname = make_filename(date_str, hour, sec)

                    def p(folder):
                        return f"{folder}/{fname}"

                    if p("MON-SPAN") in all_files:
                        d = read_json_from_zip(zf, p("MON-SPAN"))
                        if d: b_spec1[sec], b_spec2[sec], b_span[sec] = parse_mon_span(d)

                    if p("NAV-CLOCK") in all_files:
                        d = read_json_from_zip(zf, p("NAV-CLOCK"))
                        if d: b_clock[sec] = parse_scalar(d, CLOCK_FIELDS)

                    if p("NAV-DOP") in all_files:
                        d = read_json_from_zip(zf, p("NAV-DOP"))
                        if d: b_dop[sec] = parse_scalar(d, DOP_FIELDS)

                    if p("NAV-POSECEF") in all_files:
                        d = read_json_from_zip(zf, p("NAV-POSECEF"))
                        if d: b_pos[sec] = parse_scalar(d, POSECEF_FIELDS)

                    if p("NAV-PVT") in all_files:
                        d = read_json_from_zip(zf, p("NAV-PVT"))
                        if d: b_pvt[sec] = parse_scalar(d, PVT_FIELDS)

                    if p("NAV-SAT") in all_files:
                        d = read_json_from_zip(zf, p("NAV-SAT"))
                        if d: b_sat[sec] = parse_nav_sat(d)

                    if p("RXM-RAWX") in all_files:
                        d = read_json_from_zip(zf, p("RXM-RAWX"))
                        if d: b_rawx[sec] = parse_rxm_rawx(d)

            end = cursor + SECONDS_PER_HOUR
            ds_spec1[cursor:end] = b_spec1
            ds_spec2[cursor:end] = b_spec2
            ds_span [cursor:end] = b_span
            ds_clock[cursor:end] = b_clock
            ds_dop  [cursor:end] = b_dop
            ds_pos  [cursor:end] = b_pos
            ds_pvt  [cursor:end] = b_pvt
            ds_sat  [cursor:end] = b_sat
            ds_rawx [cursor:end] = b_rawx
            ds_label[cursor:end] = label
            ds_hour [cursor:end] = hour
            ds_day  [cursor:end] = day.encode()
            cursor += SECONDS_PER_HOUR

        print(f"  Day {day} done — {cursor:,} total samples")

print(f"\nDone! {OUTPUT_HDF5} written with {cursor:,} samples")