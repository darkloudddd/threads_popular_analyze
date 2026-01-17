import argparse
import os
import yaml
from crawler import Crawler
from storage import Storage
from playwright.sync_api import sync_playwright

def _resolve_path(base_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)

def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Threads crawler (manual login + scroll collection).")
    ap.add_argument("--start-url", default=None, help="start URL for the feed")
    ap.add_argument("--max-scrolls", type=int, default=None, help="max scroll iterations")
    ap.add_argument("--pause", type=float, default=None, help="seconds to pause between scrolls")
    ap.add_argument("--output", default="result.txt", help="output file name or path")
    ap.add_argument("--user-data", default="user_data", help="path for persistent browser profile")
    ap.add_argument("--debug-dir", default="debug_html", help="path for debug HTML snapshots")
    ap.add_argument("--append", action="store_true", help="append to output instead of overwrite")
    ap.add_argument("--config", default="config/config.yaml", help="path to config file")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Load config
    config_path = _resolve_path(script_dir, args.config)
    config = load_config(config_path)
    
    # Resolve values (CLI > Config > Defaults)
    start_url = args.start_url or config.get("urls", {}).get("start", "https://www.threads.net/")
    max_scrolls = args.max_scrolls or config.get("settings", {}).get("max_scrolls", 500)
    pause = args.pause or config.get("settings", {}).get("scroll_pause_time", 1.0)
    
    user_data_dir = _resolve_path(project_root, args.user_data)
    debug_dir = _resolve_path(project_root, args.debug_dir)
    output_path = _resolve_path(project_root, args.output)

    print(f"Launching browser for manual login (will use folder '{user_data_dir}').")
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            viewport=config.get("settings", {}).get("viewport", {"width":1200,"height":900}),
        )
        page = ctx.new_page()
        page.goto(config.get("urls", {}).get("login", "https://www.threads.net/login"))
        input("Perform manual login in the opened browser window, then press Enter here to start collection...")
        ctx.close()

    storage = Storage(output_path)
    crawler = Crawler(config=config, login_handler=None, storage=storage)
    print("Beginning automated scroll & collection...")
    texts = crawler.crawl_and_collect(
        start_url=start_url,
        max_scrolls=max_scrolls,
        pause=pause,
        user_data_dir=user_data_dir,
        debug_dir=debug_dir,
    )
    print(f"Collected {len(texts)} text blocks. Saving to {output_path}")
    storage.save_results(output_path, texts, append=args.append)
    print("Done. Output written.")

if __name__ == "__main__":
    main()
