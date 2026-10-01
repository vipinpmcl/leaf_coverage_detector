#!/usr/bin/env python3
"""Download openly licensed grass/field background images from Wikimedia Commons.

Usage:
    python download_grass_backgrounds.py --output-dir data/grass_backgrounds --limit 200

The script uses the Wikimedia Commons API, saves source/license metadata in
manifest.jsonl, and skips files already present in the destination folder.
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "GrassBackgroundDatasetDownloader/1.0 (research dataset; contact: replace-with-your-email)"
SEARCH_TERMS = (
    'grass field landscape',
    'grass meadow landscape',
    'lawn landscape',
    'pasture field landscape',
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def api_get(params):
    url = API_URL + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def safe_filename(title, url):
    name = Path(urlparse(url).path).name or title.rsplit(":", 1)[-1]
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")
    if not Path(name).suffix:
        name += ".jpg"
    return name[:180]


def search_images(term, remaining):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": term, "gsrnamespace": 6, "gsrlimit": min(50, remaining),
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1600,
    }
    data = api_get(params)
    pages = data.get("query", {}).get("pages", {}).values()
    return list(pages)


def download(url, destination, max_bytes):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError(f"File exceeds size limit ({int(length)} bytes)")
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"Unexpected content type: {content_type}")
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("File exceeds size limit")
    if not data:
        raise ValueError("Empty download")
    destination.write_bytes(data)


def metadata_value(extmetadata, key):
    item = extmetadata.get(key, {})
    return re.sub(r"<[^>]+>", "", item.get("value", "")).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/grass_backgrounds"))
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of images to download")
    parser.add_argument("--max-mb", type=int, default=20, help="Maximum size per image (MB)")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between downloads")
    parser.add_argument("--email", help="Contact email included in the API User-Agent")
    args = parser.parse_args()
    if args.limit < 1 or args.max_mb < 1:
        parser.error("--limit and --max-mb must be positive")

    global USER_AGENT
    if args.email:
        USER_AGENT = f"GrassBackgroundDatasetDownloader/1.0 (research dataset; {args.email})"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    seen_titles = set()
    downloaded = 0
    max_bytes = args.max_mb * 1024 * 1024

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for term in SEARCH_TERMS:
            if downloaded >= args.limit:
                break
            print(f"Searching Commons: {term}")
            try:
                pages = search_images(term, args.limit - downloaded)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"  Search failed: {exc}")
                continue

            for page in pages:
                if downloaded >= args.limit:
                    break
                title = page.get("title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                info = (page.get("imageinfo") or [{}])[0]
                url = info.get("thumburl") or info.get("url", "")
                suffix = Path(urlparse(url).path).suffix.lower()
                if not url or suffix not in IMAGE_EXTENSIONS:
                    continue
                filename = safe_filename(title, url)
                destination = args.output_dir / filename
                if destination.exists():
                    continue
                ext = info.get("extmetadata", {})
                try:
                    download(url, destination, max_bytes)
                except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                    print(f"  Skipped {title}: {exc}")
                    continue

                record = {
                    "filename": filename,
                    "title": title,
                    "source_page": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
                    "file_url": info.get("descriptionurl", url),
                    "license": metadata_value(ext, "LicenseShortName"),
                    "license_url": metadata_value(ext, "LicenseUrl"),
                    "artist": metadata_value(ext, "Artist"),
                    "query": term,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                downloaded += 1
                print(f"  [{downloaded}/{args.limit}] {filename}")
                time.sleep(max(0, args.delay))

    print(f"Downloaded {downloaded} images into {args.output_dir}")
    print(f"Attribution and license details: {manifest_path}")


if __name__ == "__main__":
    main()
