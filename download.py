# download_eci_ac_pc_series.py
import time
import argparse
from pathlib import Path
from urllib.parse import urlparse
import requests

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Referer": "https://www.eci.gov.in/",
    "Origin": "https://www.eci.gov.in",
    "Accept": "application/pdf",
    "Accept-Language": "en-US,en;q=0.9",
}


def eci_url(
    ac: int,
    pc: int,
    series: str = "S04",
    path: str = "ASD",
    host: str = "www.eci.gov.in",
) -> str:
    # https://{host}/eci-backend/public/ER/{series}/{path}/{ac}/AC{ac}P{pc}.pdf
    return f"https://{host}/eci-backend/public/ER/{series}/{path}/{ac}/AC{ac}P{pc}.pdf"


def download_ac_range(
    ac_start: int,
    ac_end: int,
    max_pc_404: int = 10,  # <-- stop after this many *consecutive* 404s within an AC
    pause_sec: float = 0.4,
    series: str = "S04",  # <-- change for a new series segment after /ER/
    path: str = "ASD",  # <-- change for a new path segment after series/
    pc_start: int = 1,  # <-- start PC number (for the first AC only); later ACs reset to 1
    pc_hard_cap: int | None = None,  # optional absolute ceiling; None = unlimited PCs
) -> int:
    total_downloads = 0
    bar = (
        tqdm(total=None, unit="pdf", dynamic_ncols=True, desc="Downloads")
        if tqdm
        else None
    )

    with requests.Session() as s:
        s.headers.update(HEADERS)
        try:
            s.get("https://www.eci.gov.in/", timeout=30)  # warm cookies
        except Exception:
            pass

        for ac in range(ac_start, ac_end + 1):
            misses = 0
            pc = pc_start if ac == ac_start else 1
            print(
                f"\n=== AC {ac}: scanning PCs from {pc} upward (until {max_pc_404} consecutive 404s) ==="
            )

            while True:
                if pc_hard_cap is not None and pc > pc_hard_cap:
                    print(f"AC {ac}: reached hard cap PC={pc_hard_cap}.")
                    break

                got_200 = False
                saw_404 = False

                for host in ("www.eci.gov.in", "eci.gov.in"):
                    url = eci_url(ac, pc, series=series, path=path, host=host)
                    try:
                        r = s.get(url, stream=True, timeout=60, allow_redirects=True)
                        if r.status_code == 404:
                            saw_404 = True
                            continue
                        if r.status_code == 200 and "pdf" in (
                            r.headers.get("Content-Type", "").lower()
                        ):
                            filename = Path(urlparse(url).path).name
                            with open(filename, "wb") as f:
                                for chunk in r.iter_content(64 * 1024):
                                    if chunk:
                                        f.write(chunk)
                            total_downloads += 1
                            got_200 = True
                            if bar is not None:
                                bar.update(1)
                                bar.set_postfix_str(f"AC{ac}P{pc} → {filename}")
                            print(f"pdf {total_downloads} downloaded: {filename}")
                            break
                        else:
                            print(f"AC{ac}P{pc}: HTTP {r.status_code} (skipping)")
                    except requests.RequestException as e:
                        print(f"AC{ac}P{pc}: request error: {e}")

                if got_200:
                    misses = 0
                elif saw_404:
                    misses += 1
                    if bar is not None:
                        bar.set_postfix_str(
                            f"AC{ac}P{pc} 404 (miss {misses}/{max_pc_404})"
                        )

                if misses >= max_pc_404:
                    print(
                        f"AC {ac}: stopping after {misses} consecutive 404s. Last tried PC={pc}."
                    )
                    break

                pc += 1
                time.sleep(pause_sec)

    if bar is not None:
        bar.close()
    print(f"\nAll done. Total PDFs: {total_downloads}")
    return total_downloads


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ECI PDFs by AC/PC, stopping after N consecutive 404s."
    )
    parser.add_argument(
        "--ac-start", type=int, default=2, help="Starting AC number (default: 2)"
    )
    parser.add_argument(
        "--ac-end", type=int, default=2, help="Ending AC number, inclusive (default: 2)"
    )
    parser.add_argument(
        "--max-pc-404",
        type=int,
        default=10,
        help="Consecutive 404s to stop an AC (default: 10)",
    )
    parser.add_argument(
        "--pause", type=float, default=0.4, help="Pause between tries (seconds)"
    )
    parser.add_argument(
        "--series",
        type=str,
        default="S04",
        help="Series segment after /ER/ (default: S04)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="ASD",
        help="Path segment after the series (default: ASD)",
    )
    parser.add_argument(
        "--pc-start",
        type=int,
        default=1,
        help="PC start for the first AC only (default: 1)",
    )
    parser.add_argument(
        "--pc-hard-cap",
        type=int,
        default=None,
        help="Optional absolute PC ceiling (default: None)",
    )
    args = parser.parse_args()

    download_ac_range(
        ac_start=args.ac_start,
        ac_end=args.ac_end,
        max_pc_404=args.max_pc_404,
        pause_sec=args.pause,
        series=args.series,
        path=args.path,
        pc_start=args.pc_start,
        pc_hard_cap=args.pc_hard_cap,
    )
