"""Collect months of data end to end. Works on Windows, macOS and Linux.

    python scripts/download/collect.py --laptop 3
    python scripts/download/collect.py --months 2024-01 2024-04

Use this instead of the .sh scripts on Windows. Typing `bash` in PowerShell
launches WSL, which is a separate Linux system that cannot see the Windows
virtual environment -- every download then fails with "command not found".
This script runs in whatever Python you launch it with, so that cannot happen.

For each month:
  1. download all six products, subset to the domain, server-side
  2. run the data check; skip the month if it fails
  3. harmonize into one cube
  4. verify the profile actually decreases with depth
  5. stage the cube in to_upload/ and clear the raw folders

Safe to stop and rerun: months already in to_upload/ are skipped, so a closed
laptop costs you one month rather than the whole run.
"""
from __future__ import annotations

import argparse
import calendar
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

BOX = dict(minimum_longitude=45, maximum_longitude=105,
           minimum_latitude=5, maximum_latitude=30)

# Reprocessed / multi-year streams. Verified coverage:
#   OSTIA REP   1981-10 .. 2026-03      DUACS my   1993-01 .. 2026-01
#   SSS my      1993-01 .. 2024-12      GLORYS my  1993-01 .. 2026-06
# The near-real-time streams do not reach back to 2022 (OSTIA NRT starts
# 2024-01-17), and mixing streams across months would put artificial jumps into
# the training data.
COPERNICUS = [
    ("ostia",  "METOFFICE-GLO-SST-L4-REP-OBS-SST", "analysed_sst", None),
    ("duacs",  "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D", "sla", None),
    ("sss",    "cmems_obs-mob_glo_phy-sss_my_multi_P1D", "sos", None),
    # thetao only, capped at 1100 m: the file carries ~10 variables and 50 levels
    ("glorys", "cmems_mod_glo_phy_my_0.083deg_P1D-m", "thetao", (0, 1100)),
]
PODAAC = [
    ("oscar", "OSCAR_L4_OC_FINAL_V2.0"),
    ("ccmp",  "CCMP_WINDS_10M6HR_L4_V3.1"),
]
LAPTOPS = {1: ["2022-01", "2022-04", "2022-07", "2022-10"],
           2: ["2023-01", "2023-04", "2023-07", "2023-10"],
           3: ["2024-01", "2024-04", "2024-07", "2024-10"]}


def find_exe(name):
    """Look beside this interpreter first, so an unactivated venv still works."""
    d = Path(sys.executable).parent
    for cand in (d / name, d / f"{name}.exe", d / "Scripts" / name,
                 d / "Scripts" / f"{name}.exe", d / "bin" / name):
        if cand.exists():
            return str(cand)
    return shutil.which(name)


def last_day(month):
    y, m = (int(v) for v in month.split("-"))
    return calendar.monthrange(y, m)[1]


def run_stage(script, log):
    """Run one pipeline stage, appending to the log. Returns (ok, output)."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    log.write(out)
    log.flush()
    return r.returncode == 0, out


def download_month(month, log):
    """Everything for one month. Returns a list of product names that failed."""
    import copernicusmarine as cm

    last = last_day(month)
    start, end = f"{month}-01T00:00:00", f"{month}-{last:02d}T23:59:59"
    failed = []

    for folder, ds_id, var, depth in COPERNICUS:
        out = ROOT / "data" / "raw" / folder
        out.mkdir(parents=True, exist_ok=True)
        print(f"    {folder:7s} ...", end="", flush=True)
        kw = dict(dataset_id=ds_id, variables=[var],
                  start_datetime=start, end_datetime=end,
                  output_directory=str(out), output_filename=f"{folder}_{month}.nc",
                  overwrite=True, **BOX)
        if depth:
            kw.update(minimum_depth=depth[0], maximum_depth=depth[1])
        try:
            t = time.time()
            cm.subset(**kw)
            f = out / f"{folder}_{month}.nc"
            sz = f.stat().st_size / 1e6 if f.exists() else 0
            print(f" {sz:7.1f} MB  ({time.time() - t:.0f}s)")
            log.write(f"{folder} ok {sz:.1f} MB\n")
            if sz == 0:
                failed.append(folder)
        except Exception as e:
            print(" FAILED")
            log.write(f"{folder} FAILED: {e}\n")
            failed.append(folder)

    dl = find_exe("podaac-data-downloader")
    for folder, coll in PODAAC:
        out = ROOT / "data" / "raw" / folder
        out.mkdir(parents=True, exist_ok=True)
        print(f"    {folder:7s} ...", end="", flush=True)
        if not dl:
            print(" FAILED (podaac-data-downloader not found -- "
                  "pip install podaac-data-subscriber)")
            log.write(f"{folder} FAILED: downloader not on PATH\n")
            failed.append(folder)
            continue
        cmd = [dl, "-c", coll, "-d", str(out),
               "-sd", f"{month}-01T00:00:00Z", "-ed", f"{month}-{last:02d}T23:59:59Z",
               "-b=45,5,105,30"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        log.write((r.stdout or "") + (r.stderr or ""))
        n = len(list(out.glob("*.nc")))
        if r.returncode == 0 and n:
            print(f" {n:4d} files")
        else:
            print(" FAILED")
            failed.append(folder)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop", type=int, choices=[1, 2, 3])
    ap.add_argument("--months", nargs="+")
    args = ap.parse_args()

    months = args.months or (LAPTOPS.get(args.laptop) if args.laptop else None)
    if not months:
        ap.error("give --laptop 1|2|3 or --months YYYY-MM ...")

    try:
        import copernicusmarine
    except ImportError:
        sys.exit("copernicusmarine is not installed in THIS python.\n"
                 f"  python is: {sys.executable}\n"
                 "  fix: pip install copernicusmarine 'boto3<1.36.0' podaac-data-subscriber")

    # Preflight, so a missing login shows up as one clear line rather than six
    # products each reporting 0.0 MB.
    problems = []
    try:
        if not copernicusmarine.login(check_credentials_valid=True):
            problems.append("Copernicus: not logged in."
                            "\n     fix: copernicusmarine login")
    except Exception as exc:
        problems.append(f"Copernicus login check failed: {exc}"
                        "\n     fix: copernicusmarine login")
    if not find_exe("podaac-data-downloader"):
        problems.append("podaac-data-downloader not found beside this python."
                        "\n     fix: pip install podaac-data-subscriber")
    netrc = Path.home() / ("_netrc" if os.name == "nt" else ".netrc")
    if not netrc.exists():
        problems.append(f"{netrc} missing -- PODAAC needs Earthdata credentials there."
                        "\n     it must contain three lines:"
                        "\n       machine urs.earthdata.nasa.gov"
                        "\n       login YOUR_EARTHDATA_USERNAME"
                        "\n       password YOUR_EARTHDATA_PASSWORD")
    if problems:
        print("\nCannot start:\n")
        for pr in problems:
            print("  - " + pr)
        print(f"\n  python in use: {sys.executable}")
        print("  Everything must be installed in THAT python, not another one.")
        return 1

    stage = ROOT / "to_upload"; stage.mkdir(exist_ok=True)
    logs = ROOT / "logs"; logs.mkdir(exist_ok=True)

    print("=" * 58)
    print(" OceanEmbed collection")
    print(f" months : {' '.join(months)}")
    print(f" python : {sys.executable}")
    print(f" output : {stage}")
    print("=" * 58)

    done, skipped, failed = [], [], []
    for month in months:
        out = stage / f"harmonized_{month}.nc"
        if out.exists():
            print(f"\n[{month}] already done ({out.stat().st_size/1e6:.0f} MB) -- skipping")
            skipped.append(month)
            continue

        print(f"\n{'-'*58}\n[{month}] starting  {time.strftime('%H:%M')}\n{'-'*58}")
        # truncate: a stale warning from a failed attempt must not reject a retry
        with open(logs / f"{month}.log", "w", encoding="utf-8") as log:
            for f in (ROOT / "data" / "raw").glob("*/*.nc"):
                f.unlink()

            print("  downloading")
            bad = download_month(month, log)
            if bad:
                print(f"[{month}] DOWNLOAD FAILED: {', '.join(bad)}  -- see logs/{month}.log")
                failed.append(f"{month}(download:{','.join(bad)})")
                continue

            print("  checking")
            ok, out_txt = run_stage("00_check_real_data.py", log)
            if not ok:
                print(f"[{month}] CHECK FAILED:")
                for line in out_txt.splitlines():
                    if "[FAIL]" in line:
                        print("     ", line.strip())
                failed.append(f"{month}(check)")
                continue

            print("  harmonizing")
            ok, out_txt = run_stage("01_harmonize_real.py", log)
            cube = ROOT / "data" / "processed" / "harmonized.nc"
            if not ok or not cube.exists():
                print(f"[{month}] HARMONIZE FAILED -- see logs/{month}.log")
                failed.append(f"{month}(harmonize)")
                continue
            if "WARNING: temperature INCREASES" in out_txt:
                print(f"[{month}] BAD PROFILE -- temperature rises with depth. Not uploading.")
                failed.append(f"{month}(profile)")
                continue

            shutil.move(str(cube), str(out))
            for f in (ROOT / "data" / "raw").glob("*/*.nc"):
                f.unlink()
            print(f"[{month}] done -> to_upload/harmonized_{month}.nc "
                  f"({out.stat().st_size/1e6:.0f} MB)")
            for line in out_txt.splitlines():
                if line.startswith("  OK: falls from") or line.strip().startswith(("sst ", "temp ")):
                    print("     ", line.strip())
            done.append(month)

    print("\n" + "=" * 58)
    print(" SUMMARY")
    print("=" * 58)
    if done:
        print("  collected  :", " ".join(done))
    if skipped:
        print("  already had:", " ".join(skipped))
    if failed:
        print("  FAILED     :", " ".join(failed))
    files = sorted(stage.glob("*.nc"))
    print("\n  ready to upload:")
    for f in files:
        print(f"    {f.name:34s} {f.stat().st_size/1e6:6.1f} MB")
    if not files:
        print("    (nothing yet)")
    print("\n  Upload everything in to_upload/ to the shared Drive folder")
    print("  oceanembed/harmonized/  then post which months you did.")
    if failed:
        print("\n  Rerun the same command to retry only the failed months.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
