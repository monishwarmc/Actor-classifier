from pathlib import Path
from icrawler.builtin import BingImageCrawler
from icrawler.downloader import ImageDownloader

TARGET_IMAGES = 500
DATA_DIR = Path("data")

# 1. Custom Downloader to force a longer connection timeout for images
class LongTimeoutDownloader(ImageDownloader):
    def download(self, task, default_ext, timeout=15, max_retry=3, **kwargs):
        # Overrides the hardcoded default timeout to 15 seconds
        return super().download(task, default_ext, timeout=timeout, max_retry=max_retry, **kwargs)

# 2. Browser identity header to prevent HTTP 403 Forbidden drops
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

for path in DATA_DIR.iterdir():
    if not path.is_dir():
        continue

    actor = path.name.replace("_", " ")
    existing = len([f for f in path.iterdir() if f.is_file()])
    remaining = TARGET_IMAGES - existing

    if remaining <= 0:
        print(f"✓ {actor}: {existing}/{TARGET_IMAGES} (skipped)")
        continue

    print(f"\n{actor}: {existing}/{TARGET_IMAGES} (need {remaining} more)")

    keywords = [
        f"{actor} actor solo photos",
        f"{actor} south indian actor",
        f"{actor} south indian actor solo images",
        f"{actor} south indian actor solo movie still"
    ]

    per_keyword = max(1, (remaining + len(keywords) - 1) // len(keywords))

    # 3. Inject our custom downloader class into the crawler setup
    crawler = BingImageCrawler(
        downloader_cls=LongTimeoutDownloader,
        downloader_threads=4, 
        storage={"root_dir": str(path)}
    )

    # Apply authentic headers globally to the parser session
    crawler.session.headers.update(BROWSER_HEADERS)

    for keyword in keywords:
        current_count = len([f for f in path.iterdir() if f.is_file()])
        if current_count >= TARGET_IMAGES:
            break

        still_needed = TARGET_IMAGES - current_count
        print(f"  Searching: {keyword} (need {still_needed} more)")

        try:
            crawler.crawl(
                keyword=keyword,
                max_num=min(per_keyword, still_needed),
                min_size=(64, 64)
            )
        except Exception as e:
            print(f"  Warning: Search chunk bypassed due to network glitch ({e})")
            continue

    final_count = len([f for f in path.iterdir() if f.is_file()])
    print(f"  Finished {actor}: {final_count} images")

print("\nAll actors processed.")
