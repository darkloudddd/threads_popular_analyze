from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import time
import re
import json

USERNAME_TOKEN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._]{1,28}[a-z0-9])?$", re.IGNORECASE)

# JavaScript to extract post content and interaction counts (Likes, Replies, etc.)
JS_EXTRACT_POST_DATA = """
(nodes) => {
  const unprocessed = nodes.filter(n => !n.dataset.processed);
  return unprocessed.map(node => {
    node.dataset.processed = "true";
    
    const getInteractionCount = (container, testId) => {
      const el = container.querySelector(`[data-testid='${testId}']`);
      if (!el) return 0;
      const text = el.innerText.replace(/[^0-9KkMm.]/g, '');
      if (!text) return 0;
      let val = parseFloat(text);
      if (text.toLowerCase().includes('k')) val *= 1000;
      if (text.toLowerCase().includes('m')) val *= 1000000;
      return Math.floor(val);
    };

    // Extract main text
    const spans = Array.from(node.querySelectorAll("span[dir='auto']"));
    const filteredTexts = spans.filter(s => {
      const t = (s.textContent || "").trim();
      if (!t) return false;
      const a = s.closest("a");
      if (a) {
        const href = a.getAttribute("href") || "";
        if (href.startsWith("/@")) return false; 
        if (href.startsWith("/search")) return false;
        if (href.startsWith("/activity")) return false;
      }
      if (s.closest("time")) return false;
      if (s.closest("button")) return false;
      return true;
    }).map(s => s.textContent.trim());

    return {
      text: filteredTexts.join(" "),
      likes: getInteractionCount(node, 'post-like-button') || 0,
      replies: getInteractionCount(node, 'post-reply-button') || 0,
      reposts: getInteractionCount(node, 'post-repost-button') || 0,
      url: node.querySelector("a[href*='/post/']")?.getAttribute("href") || ""
    };
  });
}
"""

class Crawler:
    def __init__(self, config, storage=None):
        self.config = config
        self.storage = storage
        
        self.noise_lines = set(self.config.get("filtering", {}).get("noise_lines", []))
        self.noise_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.config.get("filtering", {}).get("noise_patterns", [])
        ]
        
        self.post_selectors = self.config.get("selectors", {}).get("post_priority", [])
        self.post_container = self.config.get("selectors", {}).get("post_container", "div[data-pressable-container='true']")
        self.min_block_chars = self.config.get("filtering", {}).get("min_block_chars", 10)

    def _needs_login(self, page):
        """
        Improved login detection: check URL, title, and presence of login-related elements.
        """
        try:
            # Give the page a bit more time for initial load or redirect
            time.sleep(5)
            url = page.url.lower()
            title = page.title().lower()
            
            # 1. URL or Title check
            if "login" in url or "login" in title or "登入" in title:
                return True
                
            # 2. Check for common login button or prompt
            # Threads login page indicators
            login_selectors = [
                "input[name='username']",
                "input[name='password']",
                "button:has-text('Log in')",
                "button:has-text('登入')",
                "div:has-text('繼續使用 Facebook 登入')",
                "div:has-text('用 Instagram 登入')",
            ]
            
            for sel in login_selectors:
                if page.locator(sel).count() > 0:
                    return True

            # 3. Final check: if we see 0 posts, we are probably on a splash screen or not logged in
            post_count = page.locator(self.post_container).count()
            if post_count == 0:
                return True
                
        except Exception as e:
            # If we error during detection, safer to assume we might need login or just continue
            pass
        return False

    def _clean_text(self, text):
        if not text: return ""
        text = text.replace("\u00a0", " ")
        lines = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or any(rgx.fullmatch(ln.lower()) for rgx in self.noise_patterns):
                continue
            lines.append(ln)
        return " ".join(lines).strip()

    def crawl_and_collect(self, start_url="https://www.threads.net/", max_scrolls=200, pause=1.5, user_data_dir="user_data", debug_dir="debug_html"):
        """
        Synchronous crawling of one URL.
        """
        os.makedirs(debug_dir, exist_ok=True)
        results = []
        uncommitted_batch = []
        seen = set()

        try:
            with sync_playwright() as p:
                ctx = p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir, 
                    headless=False, 
                    viewport={"width":1200, "height":900}
                )
                page = ctx.new_page()
                page.goto(start_url)
                
                # Give it a bit more time for redirects/popups
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except:
                    pass
                
                # Manual login check
                if self._needs_login(page):
                    print("\n[!] Login required. Please login in the opened browser window.")
                    print("[*] Waiting for you to finish logging in...")
                    input(">>> After logging in, press Enter here to continue...")
                    page.goto(start_url)
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except:
                        pass
                
                print("\n" + "="*50)
                print("[*] 瀏覽器已就緒！")
                print("[*] 現在您可以：手動關閉彈窗、確認登入狀態、或是調整頁面內容。")
                print("[*] 準備好開始抓取後，請回到此視窗按下 Enter 鍵。")
                print("="*50)
                input("\n>>> [按 Enter 開始抓取資料]...")
                print("\n[*] 正在開始捲動與採集，請勿關閉瀏覽器視窗...")

                for i in tqdm(range(max_scrolls), desc="scrolling"):
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    page.wait_for_timeout(pause * 1000)

                    # Extract data
                    posts = page.locator(self.post_container).evaluate_all(JS_EXTRACT_POST_DATA)
                    
                    new_count = 0
                    for p_data in posts:
                        p_data['text'] = self._clean_text(p_data['text'])
                        if len(p_data['text']) < self.min_block_chars:
                            continue
                            
                        key = p_data['text'].strip().lower()
                        if key not in seen:
                            seen.add(key)
                            results.append(p_data)
                            uncommitted_batch.append(p_data)
                            new_count += 1
                    
                    if i > 0 and i % 20 == 0:
                        print(f" Scroll {i}: total {len(results)} posts collected.")
                        # Progressive save every 20 scrolls
                        if self.storage and uncommitted_batch:
                            self.storage.save_results(uncommitted_batch, path="result.jsonl", append=True)
                            uncommitted_batch = []
                
                # Final save for any remaining posts
                if self.storage and uncommitted_batch:
                    self.storage.save_results(uncommitted_batch, path="result.jsonl", append=True)
                    uncommitted_batch = []

                ctx.close()
        except Exception as e:
            print(f"[!] Crawler error: {e}")

        return results
