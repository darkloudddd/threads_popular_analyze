from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import time
import re

USERNAME_TOKEN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._]{1,28}[a-z0-9])?$", re.IGNORECASE)


JS_EXTRACT_TEXT = """
(nodes) => nodes.map(node => {
  const clone = node.cloneNode(true);
  clone.querySelectorAll("button, [role='button'], svg, path, time, header, footer").forEach(n => n.remove());
  return clone.innerText || "";
})
"""

JS_EXTRACT_POST_TEXT = """
(nodes) => nodes.map(node => {
  const spans = Array.from(node.querySelectorAll("span[dir='auto']"));
  const texts = spans.filter(s => {
    const t = (s.textContent || "").trim();
    if (!t) return false;
    const a = s.closest("a");
    if (a) {
      const href = a.getAttribute("href") || "";
      if (href.startsWith("/@")) return false;
      if (href.startsWith("/search")) return false;
      if (href.startsWith("/activity")) return false;
      if (href.startsWith("/following")) return false;
      if (href.startsWith("/for_you")) return false;
    }
    if (s.closest("time")) return false;
    return true;
  }).map(s => s.textContent.trim());
  return texts.join(" ");
})
"""

class Crawler:
    def __init__(self, config, login_handler=None, storage=None):
        self.config = config
        self.login_handler = login_handler
        self.storage = storage
        
        # Compile patterns from config
        self.noise_lines = set(self.config.get("filtering", {}).get("noise_lines", []))
        self.noise_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.config.get("filtering", {}).get("noise_patterns", [])
        ]
        
        self.post_selectors = self.config.get("selectors", {}).get("post_priority", [])
        self.post_container = self.config.get("selectors", {}).get("post_container", "div[data-pressable-container='true']")
        self.strip_selectors = self.config.get("selectors", {}).get("strip", "")
        self.min_block_chars = self.config.get("filtering", {}).get("min_block_chars", 10)



    def start_crawling(self):
        if self.login_handler and getattr(self.login_handler, "perform_login", None):
            if self.login_handler.perform_login():
                print("Login successful. Starting to crawl...")
                collected_words = self.collect_words()
                if self.storage:
                    self.storage.save_to_file(collected_words)
            else:
                print("Login failed. Cannot start crawling.")
        else:
            print("No automated login handler provided. Use manual login + crawl_and_collect().")

    def collect_words(self):
        # fallback/simulated collection when not using Playwright
        words = ["example", "test", "web", "spider", "example", "data", "collection"]
        print("Collected words:", words)
        return words

    def _looks_like_username(self, token):
        if not token or token.startswith("#"):
            return False
        if token.startswith("@"):
            token = token[1:]
        if not USERNAME_TOKEN_RE.fullmatch(token):
            return False
        return any(ch in token for ch in "._") or any(ch.isdigit() for ch in token)

    def _strip_leading_username(self, line):
        parts = line.split()
        if not parts:
            return ""
        if self._looks_like_username(parts[0]):
            return " ".join(parts[1:]).strip()
        return line

    def _needs_login(self, page):
        try:
            title = page.title().lower()
            if "login" in title or "登入" in title:
                return True
        except Exception:
            pass
        try:
            if page.locator("input[type='password']").count() > 0:
                return True
        except Exception:
            pass
        return False

    def _wait_for_posts(self, page, timeout_ms=20000):
        for sel in self.post_selectors:
            try:
                page.wait_for_selector(sel, timeout=timeout_ms)
                return True
            except Exception:
                continue
        return False

    def _is_noise_line(self, line):
        if re.fullmatch(r"[\W_]+", line):
            return True
        line_low = line.strip().lower()
        if line_low in self.noise_lines:
            return True
        for rgx in self.noise_patterns:
            if rgx.fullmatch(line_low):
                return True
        return False

    def _clean_block(self, text):
        if not text:
            return ""
        text = text.replace("\u00a0", " ")
        lines = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if self._is_noise_line(ln):
                continue
            ln = self._strip_leading_username(ln)
            if not ln or self._is_noise_line(ln):
                continue
            lines.append(ln)
        text = " ".join(lines)
        text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = self._strip_leading_username(text)
        return text

    def _dedupe_key(self, text):
        return re.sub(r"\s+", " ", text).strip().lower()

    def extract_text_blocks(self, html):
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.select(self.strip_selectors):
            tag.decompose()
        # try common post-like selectors first
        candidates = []
        for sel in self.post_selectors:
            candidates = soup.select(sel)
            if candidates:
                break
        texts = []
        if candidates:
            for c in candidates:
                txt = c.get_text(separator=" ", strip=True)
                if txt:
                    texts.append(txt)
        cleaned = []
        for t in texts:
            t = self._clean_block(t)
            if t and len(t) >= self.min_block_chars:
                cleaned.append(t)
        return cleaned

    def extract_text_blocks_from_page(self, page):
        try:
            post_blocks = page.locator(self.post_container).evaluate_all(JS_EXTRACT_POST_TEXT)
        except Exception:
            post_blocks = []
        cleaned = []
        for b in post_blocks:
            b = self._clean_block(b)
            if b and len(b) >= self.min_block_chars:
                cleaned.append(b)
        if cleaned:
            return cleaned
        for sel in self.post_selectors:
            try:
                blocks = page.locator(sel).evaluate_all(JS_EXTRACT_TEXT)
            except Exception:
                blocks = []
            cleaned = []
            for b in blocks:
                b = self._clean_block(b)
                if b and len(b) >= self.min_block_chars:
                    cleaned.append(b)
            if cleaned:
                return cleaned
        return self.extract_text_blocks(page.content())

    def crawl_and_collect(self, start_url="https://www.threads.net/", max_scrolls=200, pause=1.0, user_data_dir="user_data", debug_dir="debug_html"):
        """
        Opens a headful Chromium using the provided user_data_dir (so manual login persists),
        scrolls the page and collects unique text blocks. Returns a list of collected blocks.
        This version writes debug HTML files when no blocks are found to help inspection.
        """
        os.makedirs(debug_dir, exist_ok=True)
        results = []
        seen = set()
        try:
            with sync_playwright() as p:
                ctx = p.chromium.launch_persistent_context(user_data_dir=user_data_dir, headless=False, viewport={"width":1200,"height":900})
                page = ctx.new_page()
                page.goto(start_url)
                # wait for the feed to load (give it time) and for network to settle
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    time.sleep(2)
                if self._needs_login(page):
                    print("Login required. Please login in the opened browser window.")
                    input("After logging in, press Enter here to continue...")
                    try:
                        page.wait_for_load_state("networkidle", timeout=20000)
                    except Exception:
                        time.sleep(2)
                    page.goto(start_url)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        time.sleep(2)
                self._wait_for_posts(page, timeout_ms=20000)

                # report initial counts of common elements
                try:
                    art_count = page.locator("article").count()
                except Exception:
                    art_count = None
                print(f"Initial article count (page.locator('article')) = {art_count}")

                for i in tqdm(range(max_scrolls), desc="scrolling"):
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    # small pause to let new content render
                    time.sleep(pause)

                    # try waiting briefly for dynamic content
                    try:
                        page.wait_for_timeout(200)  # short wait
                    except Exception:
                        pass

                    blocks = self.extract_text_blocks_from_page(page)

                    new_count = 0
                    for b in blocks:
                        key = self._dedupe_key(b)
                        if key not in seen:
                            seen.add(key)
                            results.append(b)
                            new_count += 1

                    if new_count:
                        print(f"Scroll {i+1}: found {new_count} new blocks (total {len(results)})")
                    else:
                        # no new blocks this scroll; save a debug snapshot for inspection
                        html = page.content()
                        debug_file = os.path.join(debug_dir, f"snapshot_scroll_{i+1}.html")
                        with open(debug_file, "w", encoding="utf-8") as fh:
                            fh.write(html)
                        # also report counts of likely elements from the live page
                        try:
                            a_count = page.locator("article").count()
                            div_role_article = page.locator("div[role='article']").count()
                            data_testid = page.locator("[data-testid='post']").count()
                        except Exception:
                            a_count = div_role_article = data_testid = "n/a"
                        try:
                            page_title = page.title()
                            page_url = page.url
                        except Exception:
                            page_title = page_url = "n/a"
                        print(f"Scroll {i+1}: no new blocks — saved {debug_file} — counts: article={a_count}, div[role='article']={div_role_article}, [data-testid='post']={data_testid} — title={page_title} url={page_url}")

                ctx.close()
        except Exception as e:
            raise RuntimeError(f"crawler failed: {e}") from e

        return results
