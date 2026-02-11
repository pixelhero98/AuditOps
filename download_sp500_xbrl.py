#!/usr/bin/env python3
"""
Download S&P 500 XBRL ZIP files with resume capability.

Usage:
    python download_sp500_xbrl.py                         # Download 10-K files
    python download_sp500_xbrl.py --limit 10              # Download first 10 companies
    python download_sp500_xbrl.py --filing-type 10-Q      # Download 10-Q (quarterly)
    python download_sp500_xbrl.py --filing-type both      # Download both 10-K and 10-Q
    python download_sp500_xbrl.py --output ./my_xbrl      # Specify output directory

Filing Types:
    10-K: Annual report (1 per year per company)
    10-Q: Quarterly report (3 per year per company, Q1-Q3 only)
    both: Download all 10-K and 10-Q filings

Requirements:
    pip install requests
"""

import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# fmt: off
# S&P 500 ticker to CIK mapping https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
# https://www.sec.gov/file/company-tickers
TICKER_CIK_MAP = {
    "AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "GOOG": 1652044, "GOOGL": 1652044,
    "AMZN": 1018724, "META": 1326801, "TSLA": 1318605, "BRK.B": 1067983, "JNJ": 200406,
    "V": 1403161, "WMT": 104169, "JPM": 19617, "PG": 80424, "MA": 1141391,
    "AVGO": 1410128, "HD": 354950, "COST": 909832, "MCD": 63908, "CRM": 1108772,
    "DIS": 731697, "NFLX": 1065280, "ADBE": 796343, "AMD": 2488, "QCOM": 804842,
    "IBM": 51143, "BA": 12927, "GS": 886674, "MU": 723125, "INTC": 50104,
    "PYPL": 1633917, "TXN": 97476, "INTU": 896878, "ISRG": 1035267, "KO": 21344,
    "PEP": 884996, "PFE": 78003, "XOM": 34088, "ACN": 1467373, "APD": 2969,
    "AMAT": 6951, "MRK": 310158, "PM": 1413250, "PLD": 1045609, "RTX": 1047122,
    "SPGI": 1616707, "UBER": 1543151, "PGR": 354516, "CSCO": 858877, "NKE": 320187,
    "SYK": 310567, "NOW": 1411970, "LLY": 59478, "LOW": 60086, "NSC": 1070501,
    "UPS": 1090727, "ITW": 49826, "KDP": 1393612, "LRCX": 1038724, "LMT": 60086,
    "LVS": 100104, "MAR": 1048972, "MAS": 62996, "MMC": 62457, "MAT": 63276,
    "MET": 63908, "MTN": 1109357, "MCHP": 716371, "MNST": 1470705, "MCO": 1156039,
    "MPWR": 1609711, "MPC": 774521, "MSI": 63989, "MRNA": 1682701, "MRO": 101778,
    "MSM": 63276, "MTR": 1468242, "NCLH": 1513761, "NDAQ": 1120193, "NEE": 753165,
    "NEM": 1164727, "NLOK": 849399, "NOC": 1133421, "NOV": 816469, "NRG": 809268,
    "NDSN": 1289817, "NTAP": 1002047, "NTES": 1614715, "NVR": 925278, "NWS": 1208460,
    "NWSA": 1208460, "OMC": 29989, "ONL": 1769394, "ORCL": 1652044, "ORLY": 882835,
    "OXY": 797468, "OKTA": 1667622, "ONE": 1711997, "OKE": 100463, "OLED": 1289308,
    "OGN": 1606540, "OLN": 753939, "OMF": 702265, "OPCH": 1464152, "OPK": 701985,
    "OPT": 1741225, "ORCL": 1652044, "OSIS": 1680948, "OTEX": 1070356, "OTIS": 1748709,
    "OUR": 1737137, "OVV": 4510, "OXM": 798022, "PAAM": 1477697, "PACB": 1412598,
    "PACK": 1524472, "PAGP": 1045609, "PAG": 1809923, "PANW": 1614316, "PAYX": 723875,
    "PAYC": 1640340, "PAY": 1738839, "PCAR": 75362, "PCBK": 823768, "PCF": 1437107,
    "PCG": 75488, "PEG": 75004, "PEB": 901146200, "PEI": 1456192, "PEN": 1141936,
    "PER": 1308995, "PERI": 700340, "PERN": 823768, "PET": 1161697, "PFG": 67204,
    "PGC": 80424, "PGEN": 1564590, "PGTI": 779473, "PHI": 1084869, "PHM": 822416,
    "PHMD": 1414171, "PKE": 1417025, "PKG": 79833, "PKI": 1094289, "PKX": 1398340,
    "PLE": 1441835, "PLTR": 1649199, "PLYM": 1414486, "PMCB": 1042530, "PMD": 1369805,
    "PMPM": 1617230, "PNC": 713676, "PNM": 1037956, "PNRG": 1109806, "PNR": 77105,
    "PNW": 73541, "PODD": 1491697, "POOL": 1018724, "POR": 1000045, "POST": 69068,
    "POTX": 1661929, "POWL": 1773913, "PPBI": 1427375, "PPG": 79879, "PPL": 69487,
    "PPSI": 1491395, "PRAA": 1398033, "PRCH": 1631394, "PRCT": 1411619, "PRDS": 1584994,
    "PRED": 1658498, "PRIM": 1394980, "PRIM.A": 1394980, "PRK": 908200, "PRLB": 1421874,
    "PRO": 854054, "PROH": 1651298, "PROM": 800996, "PROS": 1373141, "PROV": 1612356,
    "PRSK": 1449255, "PRU": 77975, "PRXL": 1732267, "PRY": 1577289, "PSA": 1393311,
    "PSB": 1599959, "PSK": 1383816, "PSX": 1534701, "PT": 1452901, "PTCT": 1363165,
    "PTEN": 1403973, "PTER": 1513761, "PGTI": 1450254, "PTI": 1451505, "PTLA": 1488754,
    "PTM": 1577289, "PTSI": 912057, "PTY": 1434379, "PUBM": 1607453, "PUK": 1112681,
    "PULL": 1571695, "PUMP": 1626207, "PUNB": 1532218, "PUNK": 1698066, "PUS": 1647099,
} 
# fmt: on

OUTPUT_DIR = Path("./xbrl_downloads")
PROGRESS_FILE = OUTPUT_DIR / ".download_progress.json"

# Create a persistent session with proper headers
SESSION = None


def get_session():
    """Get or create a persistent requests session with browser-like headers"""
    global SESSION
    if SESSION is None:
        SESSION = requests.Session()
        SESSION.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )
    return SESSION


def load_progress():
    """Load already downloaded tickers from progress file"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_progress(completed_tickers):
    """Save progress to JSON file for resume capability"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(completed_tickers, f)


def get_latest_xbrl_accession(cik, filing_type="10-K"):
    """Get latest accession number for specified filing type from SEC API

    Args:
        cik: Central Index Key
        filing_type: '10-K', '10-Q', or other form type
    """
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        session = get_session()

        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Check structure
        if "filings" not in data:
            return None

        filings = data["filings"]

        # Try 'recent' first
        if "recent" in filings:
            recent = filings["recent"]
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])

            # Get latest matching form type
            for i, form in enumerate(forms):
                if form and filing_type in form:
                    return accessions[i] if i < len(accessions) else None

        return None
    except Exception as e:
        logger.debug(f"Error fetching accession for CIK {cik}: {e}")
        return None


def download_xbrl_zip(cik, ticker, accession):
    """Download XBRL ZIP file with resume support"""
    session = get_session()

    # Format accession: remove hyphens (0000320193-26-000006 -> 000032019326000006)
    accession_formatted = accession.replace("-", "")

    # Correct URL format: /Archives/edgar/data/{cik}/{accession_formatted}/{accession}-xbrl.zip
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_formatted}/{accession}-xbrl.zip"
    filename = f"{ticker}_{accession}.zip"
    filepath = OUTPUT_DIR / filename

    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        # Download with resume
        mode = "wb"
        dl_headers = {}

        if filepath.exists():
            local_size = filepath.stat().st_size
            dl_headers["Range"] = f"bytes={local_size}-"
            mode = "ab"
            logger.info(f"  Resuming {filename}...")

        # Use GET request (HEAD requests are blocked by SEC)
        resp = session.get(
            url, headers=dl_headers, stream=True, timeout=30, allow_redirects=True
        )
        resp.raise_for_status()

        with open(filepath, mode) as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"✓ {filename}")
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.error(f"✗ {filename}: 403 Forbidden")
        else:
            logger.error(f"✗ {filename}: HTTP {e.response.status_code}")
        return False
    except Exception as e:
        logger.error(f"✗ {filename}: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download S&P 500 XBRL files with resume"
    )
    parser.add_argument("--limit", type=int, help="Limit number of companies")
    parser.add_argument(
        "--output", type=str, default="./xbrl_downloads", help="Output directory"
    )
    parser.add_argument(
        "--filing-type",
        type=str,
        default="10-K",
        choices=["10-K", "10-Q", "both"],
        help="Filing type: 10-K (annual), 10-Q (quarterly), or both",
    )
    parser.add_argument(
        "--proxy", type=str, help="HTTP proxy URL (e.g., http://proxy:8080)"
    )
    args = parser.parse_args()

    global OUTPUT_DIR, PROGRESS_FILE
    OUTPUT_DIR = Path(args.output)
    PROGRESS_FILE = OUTPUT_DIR / ".download_progress.json"

    # Configure proxy if provided
    if args.proxy:
        session = get_session()
        session.proxies.update(
            {
                "http": args.proxy,
                "https": args.proxy,
            }
        )
        logger.info(f"Using proxy: {args.proxy}\n")

    # Determine which filing types to download
    if args.filing_type == "both":
        filing_types = ["10-K", "10-Q"]
    else:
        filing_types = [args.filing_type]

    # Load progress
    completed = load_progress()
    tickers = list(TICKER_CIK_MAP.items())

    if args.limit:
        tickers = tickers[: args.limit]

    total_downloads = len(tickers) * len(filing_types)
    logger.info(
        f"Downloading {', '.join(filing_types)} XBRL for {len(tickers)} companies"
    )
    logger.info(f"Output: {OUTPUT_DIR}\n")

    download_count = 0
    for form_type in filing_types:
        for i, (ticker, cik) in enumerate(tickers, 1):
            # Use ticker_formtype as key for progress tracking
            progress_key = f"{ticker}_{form_type}"

            # Skip if already downloaded
            if progress_key in completed and completed[progress_key]:
                logger.info(
                    f"[{form_type}] [{i}/{len(tickers)}] {ticker} (already done)"
                )
                download_count += 1
                continue

            logger.info(f"[{form_type}] [{i}/{len(tickers)}] {ticker}...")

            # Get latest accession for this form type
            accession = get_latest_xbrl_accession(cik, filing_type=form_type)
            if not accession:
                logger.warning(f"  No {form_type} found")
                completed[progress_key] = False
                save_progress(completed)
                continue

            # Download
            if download_xbrl_zip(cik, ticker, accession):
                completed[progress_key] = True
                download_count += 1
            else:
                completed[progress_key] = False

            save_progress(completed)
            time.sleep(0.5)  # Rate limiting

    # Summary
    success = sum(1 for v in completed.values() if v)
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Downloaded: {success}/{total_downloads}")
    logger.info(f"Output: {OUTPUT_DIR}")

    if success == 0 and len(completed) > 0:
        logger.info(f"\nNote: If you're getting 403 Forbidden errors:")
        logger.info(f"  - Try using a proxy: --proxy http://your.proxy:8080")
        logger.info(f"  - Or use a VPN to bypass network restrictions")
        logger.info(f"  - The accession numbers above can be manually downloaded")

    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
