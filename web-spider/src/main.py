import argparse
import yaml
import os
import sys
from crawler import Crawler
from storage import save_results

def load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Threads Synchronous Crawler v3.1")
    ap.add_argument("--urls", "-u", default="https://www.threads.net/", 
                    help="Target URL to crawl (default: Threads home)")
    ap.add_argument("--max-scrolls", "-n", type=int, default=100, help="max scrolls per page")
    ap.add_argument("--output", "-o", default="result.jsonl", help="output file (default: result.jsonl)")
    ap.add_argument("--user-data", default="user_data", help="browser user data dir for login persistence")
    args = ap.parse_args()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", "config.yaml")
    
    config = load_config(config_path)
    crawler = Crawler(config)

    print(f"[*] Starting Synchronous Crawler v3.1...")
    print(f"[*] Target: {args.urls}")
    
    # Run crawler
    results = crawler.crawl_and_collect(
        start_url=args.urls, 
        max_scrolls=args.max_scrolls,
        user_data_dir=args.user_data
    )

    if results:
        # Save to JSONL (supports structured weighted analysis)
        save_results(results, args.output)
        # Also save a legacy txt version for backward compatible simple analysis if needed
        txt_output = args.output.replace(".jsonl", ".txt")
        with open(txt_output, "w", encoding="utf-8") as f:
            for p in results:
                f.write(p['text'] + "\n\n")
                
        print(f"\n[+] Success! Collected {len(results)} posts.")
        print(f"[+] Saved to {args.output} (and {txt_output})")
    else:
        print("\n[-] No posts collected.")

if __name__ == "__main__":
    main()
