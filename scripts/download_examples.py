"""Download real, openly-licensed car-damage example photos from Wikimedia Commons.

Unlike ``generate_sample_data.py`` (which draws synthetic images), this fetches
**real** crash photos that are in the public domain or under CC0 / CC BY / CC
BY-SA licenses, and writes an ``ATTRIBUTIONS.md`` so the license terms are met.

These are meant purely as sample uploads for the Streamlit app. Review each
image's license on its Commons page before reusing it elsewhere.

Usage:
    python scripts/download_examples.py [--out data/examples]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Curated Wikimedia Commons files (all public domain / CC0 / CC BY / CC BY-SA).
COMMONS_TITLES = [
    "Car crash at night.jpg",
    "ChevyBlazerAccident.jpg",
    "Car accident, 16th Street, July 9, 2025.jpg",
    "Accident on National Highway.jpg",
    "Automobile accidents 01.jpg",
    "Car crash Kluang.jpg",
    "Budapest Car Accident.jpg",
    "CAR ACCIDENT.jpg",
]
UA = "CollisionVision/0.1 (open-source car-damage demo; polite sample fetch)"
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    """Strip HTML tags/entities from a Commons metadata value."""
    return html.unescape(_TAG_RE.sub("", text or "")).strip() or "Unknown"


def _slug(title: str) -> str:
    stem = Path(title).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:40]


def fetch_metadata(titles: list[str]) -> dict:
    """Query the Commons API for image URLs + license metadata."""
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(f"File:{t}" for t in titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": "1280",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def download(url: str, dest: Path, retries: int = 4) -> bool:
    """Download ``url`` to ``dest``, retrying with backoff on transient errors."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return True
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            if attempt == retries:
                print(f"! failed {dest.name}: {exc}")
                return False
            time.sleep(2.0 * attempt)  # back off before retrying
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download licensed car-damage sample photos."
    )
    parser.add_argument("--out", default="data/examples", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("real_*"):
        stale.unlink()

    pages = (fetch_metadata(COMMONS_TITLES).get("query") or {}).get("pages") or {}
    attributions = [
        "# Example image attributions",
        "",
        "Real crash photos from Wikimedia Commons, used as demo uploads.",
        "",
    ]

    index = 0
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        thumb = info.get("thumburl")
        if not thumb:
            print(f"! no URL for {page.get('title')}")
            continue
        meta = info.get("extmetadata") or {}
        title = page.get("title", "").replace("File:", "")
        license_name = _clean((meta.get("LicenseShortName") or {}).get("value"))
        author = _clean((meta.get("Artist") or {}).get("value"))
        source_page = info.get("descriptionurl", "")

        ext = ".png" if info.get("mime") == "image/png" else ".jpg"
        filename = f"real_{index:02d}_{_slug(title)}{ext}"
        if not download(thumb, out_dir / filename):
            continue
        print(f"downloaded {filename}  [{license_name}]")

        attributions.append(
            f"- **{filename}** — “{title}” by {author}, {license_name}. {source_page}"
        )
        index += 1
        time.sleep(1.5)  # be polite to the Commons servers between downloads

    (out_dir / "ATTRIBUTIONS.md").write_text(
        "\n".join(attributions) + "\n", encoding="utf-8"
    )
    print(f"\nSaved {index} image(s) + ATTRIBUTIONS.md to {out_dir}")


if __name__ == "__main__":
    main()
