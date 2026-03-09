import argparse
import os
import re
import math
import hashlib
import sys
import time
import yaml
from collections import Counter

# --- Optional deps (jieba & pandas) ---
try:
    import jieba
    import jieba.analyse
    import jieba.posseg as pseg
    JIEBA_OK = True
except Exception:
    JIEBA_OK = False
    jieba = None
    pseg = None

try:
    import pandas as pd
except Exception:
    pd = None

# --- CKIP deps (optional) ---
try:
    from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
    CKIP_OK = True
except Exception:
    CKIP_OK = False

# POS 保留：名詞/專有名詞/地名/機構名/動名詞等
POS_KEEP_PREFIX = ("n", "nr", "ns", "nt", "nz", "vn")
# POS 加權：專有名詞、地名、機構名加權，提升趨勢感
POS_WEIGHTS = {
    "nz": 1.5,  # 其他專名
    "nt": 1.5,  # 機構名
    "ns": 1.3,  # 地名
    "nr": 1.2,  # 人名
}

USERNAME_TOKEN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._]{1,28}[a-z0-9])?$", re.IGNORECASE)

# --- 注音文 / 台灣口語轉換表 ---
ZHUYIN_MAP = {
    "ㄉ": "的", "ㄅ": "吧", "ㄇ": "嗎", "ㄋ": "呢",
    "ㄌ": "了", "ㄛ": "喔", "ㄟ": "欸", "ㄏ": "呵",
    "ㄆ": "噗", "ㄎ": "可",
}
_ZHUYIN_RE = re.compile("(" + "|".join(re.escape(k) for k in ZHUYIN_MAP) + ")")

def normalize_taiwanese_text(text: str) -> str:
    """Convert common zhuyin abbreviations and normalize spaces as punctuation."""
    # 1. 注音文轉換 (ㄉ -> 的, ㄅ -> 吧, ...)
    text = _ZHUYIN_RE.sub(lambda m: ZHUYIN_MAP[m.group()], text)
    # 2. 台灣用戶常以空白取代標點 — 將連續空白轉為全形逗號幫助 jieba 斷句
    #    但只在「中文字 空白 中文字」的情境下做替換，避免破壞英文
    text = re.sub(r'(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])', '，', text)
    return text

# --------------------------
# Helpers
# --------------------------

def load_analysis_config(config_path):
    analysis_config = {
        "noisy_ascii": set(),
        "drop_patterns": [],
        "promo_ascii": [],
        "phrase_drop_regex": [],
        "precision_stopwords": set(),
        "meaningful_ascii_keep": set(),
        "ignore_tokens": set(),
        "pos_filters": []
    }
    
    if not os.path.exists(config_path):
        print(f"[WARN] Config not found: {config_path}")
        return analysis_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            
        analysis_config["noisy_ascii"] = set(data.get("noisy_ascii", []))
        
        # Compile regex patterns
        analysis_config["drop_patterns"] = [
            re.compile(p, re.IGNORECASE) for p in data.get("drop_patterns", [])
        ]
        analysis_config["promo_ascii"] = [
            re.compile(p) for p in data.get("promo_ascii", [])
        ]
        analysis_config["phrase_drop_regex"] = [
            re.compile(p) for p in data.get("phrase_drop_regex", [])
        ]
        
        analysis_config["precision_stopwords"] = set(data.get("precision_stopwords", []))
        analysis_config["meaningful_ascii_keep"] = set(data.get("meaningful_ascii_keep", []))
        analysis_config["ignore_tokens"] = set(data.get("ignore_tokens", []))
        analysis_config["pos_filters"] = set(data.get("pos_filters", []))
        
    except Exception as e:
        print(f"[ERROR] Failed to load analysis config: {e}")
        
    return analysis_config


class CKIPAnalyzer:
    """
    Wrapper for CKIP Transformers models.
    Lazily loads models to save memory if not used.
    """
    def __init__(self, level=1, device=-1):
        """
        level: 1 (albert-tiny), 2 (albert-base), 3 (bert-base)
        device: -1 for CPU, 0+ for GPU
        """
        self.level = level
        self.device = device
        self.ws_driver = None
        self.pos_driver = None
        self.ner_driver = None

    def _load(self):
        # Map level to model name
        model_map = {
            1: "albert-tiny",
            2: "albert-base",
            3: "bert-base"
        }
        model_name = model_map.get(self.level, "albert-tiny")

        if self.ws_driver is None:
            print(f"[INFO] Loading CKIP WordSegmenter (model: {model_name}, device: {self.device})...")
            self.ws_driver = CkipWordSegmenter(model=model_name, device=self.device)
        if self.pos_driver is None:
            print(f"[INFO] Loading CKIP PosTagger (model: {model_name})...")
            self.pos_driver = CkipPosTagger(model=model_name, device=self.device)
        if self.ner_driver is None:
            print(f"[INFO] Loading CKIP NerChunker (model: {model_name})...")
            self.ner_driver = CkipNerChunker(model=model_name, device=self.device)

    def analyze(self, posts):
        """
        Batch analyze posts.
        Returns: list of (words, pos, ner) per post
        """
        self._load()
        print(f"[INFO] CKIP Analyzing {len(posts)} posts...")
        # CKIP batch processing
        ws_results = self.ws_driver(posts)
        pos_results = self.pos_driver(ws_results)
        ner_results = self.ner_driver(posts) # NER takes raw text list
        
        return list(zip(ws_results, pos_results, ner_results))


def _render_progress(label, idx, total, start_time):
    if total <= 0:
        return
    width = 28
    pct = idx / total
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    rate = idx / elapsed if elapsed > 0 else 0.0
    eta = (total - idx) / rate if rate > 0 else 0.0
    sys.stderr.write(
        f"\r{label} [{bar}] {idx}/{total} {pct*100:5.1f}% {rate:5.1f}/s ETA {eta:4.0f}s"
    )
    sys.stderr.flush()


def iter_with_progress(items, label="processing", enable=True):
    if not enable:
        for item in items:
            yield item
        return
    total = len(items)
    if total <= 0:
        return
    start_time = time.time()
    update_every = max(1, total // 100)
    for idx, item in enumerate(items, 1):
        if idx == 1 or idx == total or idx % update_every == 0:
            _render_progress(label, idx, total, start_time)
        yield item
    sys.stderr.write("\n")
    sys.stderr.flush()


def looks_like_username(token: str) -> bool:
    if not token:
        return False
    if token.startswith("#"):
        return False
    if token.startswith("@"):
        token = token[1:]
    if not USERNAME_TOKEN_RE.fullmatch(token):
        return False
    return "." in token or "_" in token


def normalize_text_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_stopwords(custom_path, default_path, noisy_ascii):
    s = set()
    # Load default stopwords
    if default_path and os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    s.add(w)
    
    # Load custom stopwords
    if custom_path and os.path.exists(custom_path):
        with open(custom_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    s.add(w)
                    
    if noisy_ascii:
        s |= noisy_ascii
    return s


def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    try:
        text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    except re.error:
        pass
    # 台灣文本正規化 (注音文轉換 + 空白轉標點)
    text = normalize_taiwanese_text(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def strip_threads_metadata(text: str) -> str:
    """
    Remove Threads UI artefacts that get captured by the crawler:
    - Time prefix: '5小時', '23分鐘', '1天', '2天'
    - '第一則串文' prefix
    - Trailing engagement metrics: '1,411 47 13 21', '2.5 萬 211 1,234'
    - Merged/doubled numbers from dynamic UI: '502503'
    - '劇透' tag spam
    - '翻譯' / 'AI 資訊' suffix
    - Countdown timer: '還剩N小時'
    """
    # 1. 移除開頭的 '第一則串文'
    text = re.sub(r'^第一則串文\s*', '', text)
    # 2. 移除開頭的時間前綴 (5小時, 23分鐘, 1天, 22小時)
    text = re.sub(r'^\d+\s*(小時|分鐘|天)\s*', '', text)
    # 3. 移除末尾 '還剩N小時' 之類的倒計時
    text = re.sub(r'\s*還剩\d+小時\s*$', '', text)
    # 4. 移除末尾的 '翻譯' / 'AI 資訊'
    text = re.sub(r'\s*(翻譯|AI\s*資訊)\s*$', '', text)
    # 5. 移除尾部的互動數字 (likes, comments, shares, reposts)
    #    常見格式: '1,411 47 13 21' 或 '2.5 萬 211 1,234 6,381'
    #    也有合併格式: '502503 16 1 16'
    text = re.sub(r'\s+[\d,.]+(\s*萬)?(?:\s+[\d,.]+(?:\s*萬)?){1,5}\s*$', '', text)
    # 6. 移除重複的 '劇透' (保留第一個)
    text = re.sub(r'(劇透\s*){2,}', '劇透 ', text)
    text = text.strip()
    return text


def split_posts(text, drop_patterns):
    """
    Split text into list of post strings.
    Assume each post is separated by one or more blank lines.
    Filter out entire posts if they match any 'drop_pattern'.
    Also removes repetitive noise lines (like '劇透 劇透...').
    """
    raw_posts = re.split(r'\n\s*\n', text)
    
    # Fallback if double-newline splitting yields too few posts for a large file
    if len(raw_posts) < 10 and len(text) > 2000:
        raw_posts = text.splitlines()

    valid_posts = []
    
    for i, p in enumerate(raw_posts):
        p = p.strip()
        if not p:
            continue
            
        # 1. Check drop patterns (entire post level)
        should_drop = False
        for pat in drop_patterns:
            if pat.search(p):
                should_drop = True
                break
        if should_drop:
            continue
            
        # 2. Heuristic: Remove repetitive spam lines within a post
        lines = p.split('\n')
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped: continue
            
            # Check for excessive repetition (Word Word Word...)
            if re.search(r'((\S{1,4})\s+)\1{2,}', line_stripped):
                continue
            cleaned_lines.append(line)
            
        p = "\n".join(cleaned_lines).strip()
        if not p:
            continue

        valid_posts.append(p)
    return valid_posts


def fallback_tokenize(text):
    return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_#@]+", text)


def simhash(tokens):
    if not tokens:
        return None
    v = [0] * 64
    for tok in set(tokens):
        h = hashlib.md5(tok.encode("utf-8")).digest()
        x = int.from_bytes(h[:8], "big")
        for i in range(64):
            if (x >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1
    fp = 0
    for i, val in enumerate(v):
        if val > 0:
            fp |= 1 << i
    return fp


def hamming_distance(a, b):
    return bin(a ^ b).count("1")


def is_near_duplicate(sig, buckets, threshold=3):
    if sig is None:
        return False
    mask = (1 << 16) - 1
    for band in range(4):
        key = (band, (sig >> (band * 16)) & mask)
        for other in buckets.get(key, []):
            if hamming_distance(sig, other) <= threshold:
                return True
    for band in range(4):
        key = (band, (sig >> (band * 16)) & mask)
        buckets.setdefault(key, []).append(sig)
    return False


def tokenize_post(text, keep_hash_at=False, pos_only=False, stopwords=None, min_len=2):
    tokens = []
    if JIEBA_OK:
        if pos_only:
            for w, flag in pseg.cut(text):
                ww = w.strip()
                if not ww:
                    continue
                if not keep_hash_at and (ww.startswith("#") or ww.startswith("@")):
                    continue
                if re.fullmatch(r"\d+", ww):
                    continue
                if re.fullmatch(r"[A-Za-z0-9_]+", ww):
                    ww = ww.lower()
                if not flag.startswith(POS_KEEP_PREFIX):
                    continue
                if stopwords and ww in stopwords:
                    continue
                if len(ww) < min_len:
                    continue
                tokens.append(ww)
        else:
            for w in jieba.lcut(text):
                ww = w.strip()
                if not ww:
                    continue
                if not keep_hash_at and (ww.startswith("#") or ww.startswith("@")):
                    continue
                if re.fullmatch(r"\d+", ww):
                    continue
                if re.fullmatch(r"[A-Za-z0-9_]+", ww):
                    ww = ww.lower()
                if stopwords and ww in stopwords:
                    continue
                if len(ww) < min_len:
                    continue
                tokens.append(ww)
    else:
        for w in fallback_tokenize(text):
            ww = w.strip()
            if not ww:
                continue
            if not keep_hash_at and (ww.startswith("#") or ww.startswith("@")):
                continue
            if re.fullmatch(r"\d+", ww):
                continue
            if re.fullmatch(r"[A-Za-z0-9_]+", ww):
                ww = ww.lower()
            if stopwords and ww in stopwords:
                continue
            if len(ww) < min_len:
                continue
            tokens.append(ww)
    return tokens


def ngrams(seq, n):
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i:i+n])


def extract_keywords_with_pos(posts, top_n=30, stopwords=None, ignore_tokens=None, pos_filters=None, ckip_results=None):
    """
    Use TF-IDF to find top keywords, but with POS filtering and weighting.
    ckip_results: if provided, use pre-tokenized (ws, pos, ner) instead of jieba.
    """
    if stopwords is None:
        stopwords = set()
    if ignore_tokens is None:
        ignore_tokens = set()
    if pos_filters is None:
        pos_filters = []  # Empty means no filtering
        
    doc_freq = Counter()
    total_docs = len(posts)
    post_tokens_list = [] # List of lists of (word, flag)
    
    if ckip_results:
        # Use CKIP tokens
        for ws, pos, ner in ckip_results:
            unique_tokens = set()
            tokens_for_this_post = []
            for w, flag in zip(ws, pos):
                w = w.strip().lower()
                if not w or w in stopwords or w in ignore_tokens:
                    continue
                if len(w) < 2 and not re.match(r'^[a-zA-Z0-9]+$', w):
                    continue
                
                # CKIP flags are uppercase and more detailed, but we check prefix
                if pos_filters:
                    matched_pos = False
                    f_low = flag.lower()
                    for pf in pos_filters:
                        if f_low.startswith(pf):
                            matched_pos = True
                            break
                    if not matched_pos:
                        continue
                
                unique_tokens.add(w)
                tokens_for_this_post.append((w, flag.lower()))
            for t in unique_tokens:
                doc_freq[t] += 1
            post_tokens_list.append(tokens_for_this_post)
    else:
        # Use Jieba/Pseg
        for p in posts:
            words = pseg.cut(p) if pseg else []
            unique_tokens = set()
            tokens_for_this_post = []
            
            for w, flag in words:
                w = w.strip().lower()
                if len(w) < 2 and w not in ["ai", "ui", "ux"]: 
                    if not re.match(r'^[a-zA-Z0-9]+$', w): 
                        continue

                if w in stopwords or w in ignore_tokens:
                    continue
                
                if pos_filters:
                    matched_pos = False
                    for pf in pos_filters:
                        if flag.startswith(pf):
                            matched_pos = True
                            break
                    if not matched_pos:
                        continue
                
                unique_tokens.add(w)
                tokens_for_this_post.append((w, flag))
                
            for t in unique_tokens:
                doc_freq[t] += 1
            post_tokens_list.append(tokens_for_this_post)

    # Calculate TF-IDF with POS Weighting
    total_term_freq = Counter()
    term_pos_map = {} # Cache POS for weighting (use most frequent flag for word)
    
    for tokens_info in post_tokens_list:
        for w, flag in tokens_info:
            total_term_freq[w] += 1
            if w not in term_pos_map:
                term_pos_map[w] = flag
        
    tfidf_scores = {}
    for term, count in total_term_freq.items():
        tf = count
        df = doc_freq[term]
        idf = math.log(total_docs / (df + 1))
        
        # Apply POS Weighting
        weight = 1.0
        flag = term_pos_map.get(term, "n")
        for p_pref, p_weight in POS_WEIGHTS.items():
            if flag.startswith(p_pref):
                weight = max(weight, p_weight)
                break
        
        tfidf_scores[term] = tf * idf * weight
        
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]


def generate_trend_report(keywords, phrases, hashtags, ckip_results=None):
    """
    Generate a categorized summary of trends for the user.
    Safe for Windows terminals (avoiding emoji encoding issues).
    """
    header = "\n" + "="*50 + "\nTHREADS 流行趨勢洞察簡報 (Trend Insight Report)\n" + "="*50
    try:
        print(header)
    except UnicodeEncodeError:
        print(header.encode('ascii', errors='replace').decode('ascii'))

    # Categories
    people_media = []
    places = []
    food_lifestyle = []
    events_items = []
    
    # Food/Lifestyle keywords to help categorization
    FOOD_KEYWORDS = {"甜點", "咖啡", "草莓", "蛋糕", "火鍋", "宵夜", "美食", "早午餐", "餐廳", "隱藏"}
    
    # Process keywords and phrases
    seen = set()
    
    # 1. Use CKIP NER results if available for high-precision tagging
    if ckip_results:
        # ner: list of list of NerToken(word, ner, start, end)
        entity_freq = Counter()
        for doc_ws, doc_pos, doc_ner in ckip_results:
            for ent in doc_ner:
                entity_freq[(ent.word, ent.ner)] += 1
        
        # Sort entities by frequency
        sorted_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)
        for (word, etype), count in sorted_entities:
            if word in seen or len(word) < 2 or word in FOOD_KEYWORDS:
                continue
            
            if etype in ["PERSON", "WORK_OF_ART"]:
                people_media.append(f"[NER:{etype}] {word}")
                seen.add(word)
            elif etype in ["GPE", "LOC", "FAC"]:
                places.append(f"[NER:{etype}] {word}")
                seen.add(word)
            elif etype in ["PRODUCT", "ORG", "EVENT"]:
                events_items.append(f"[NER:{etype}] {word}")
                seen.add(word)

    # 2. Process phrases (Bigrams / Trigrams)
    for item in phrases[:15]: # Take top phrases
        p = item[0]
        if p in seen: continue
        if any(f in p for f in FOOD_KEYWORDS):
            food_lifestyle.append(f"[Food]  {p}")
        elif any(k in p for k in ["演出", "合唱", "電影", "進擊", "巨人", "角色", "演員", "歌手"]):
            people_media.append(f"[Media] {p}")
        else:
            events_items.append(f"[Event] {p}")
        seen.add(p)

    # 3. Process remaining Keywords
    for word, score in keywords[:60]:
        if word in seen: continue
        # Manual tagging helper (simplified)
        if any(f in word for f in FOOD_KEYWORDS):
            food_lifestyle.append(f"[Food]  {word}")
        elif any(k in word for k in ["台南", "台北", "日本", "台中", "高雄", "花蓮", "旅遊", "咖啡廳"]):
            places.append(f"[Place] {word}")
        elif len(word) >= 2:
            events_items.append(f"[Trend] {word}")

    def safe_print(msg):
        try:
            print(msg)
        except UnicodeEncodeError:
            # Fallback for Windows CP950
            print(msg.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

    safe_print("\n[人物與影視娛樂]")
    for item in people_media[:8]: safe_print(f"  - {item}")
    
    safe_print("\n[熱門地點與旅遊]")
    for item in places[:8]: safe_print(f"  - {item}")
    
    safe_print("\n[美食、甜點與生活]")
    for item in food_lifestyle[:8]: safe_print(f"  - {item}")
    
    safe_print("\n[熱門商品、事件與話題]")
    for item in events_items[:12]: safe_print(f"  - {item}")

    if hashtags:
        safe_print("\n[熱門 Hashtags]")
        tag_list = sorted(hashtags.items(), key=lambda x: x[1], reverse=True)
        tags_line = "  " + " ".join([h for h, count in tag_list[:6]])
        safe_print(tags_line)
    
    safe_print("\n" + "="*50)


def compute_tfidf(all_docs_tokens):
    N = len(all_docs_tokens)
    df = Counter()
    tf_per_doc = []
    for toks in all_docs_tokens:
        tf = Counter(toks)
        tf_per_doc.append(tf)
        df.update(set(tf.keys()))
    tfidf = Counter()
    for tf in tf_per_doc:
        for w, cnt in tf.items():
            tf_val = 1.0 + math.log(cnt)
            idf = math.log((N + 1) / (df[w] + 1)) + 1.0
            tfidf[w] += tf_val * idf
    return tfidf


def compute_tfidf_binary(all_docs_tokens):
    N = len(all_docs_tokens)
    df = Counter()
    for toks in all_docs_tokens:
        df.update(set(toks))
    tfidf = Counter()
    for toks in all_docs_tokens:
        seen = set()
        for w in toks:
            if w in seen:
                continue
            seen.add(w)
            idf = math.log((N + 1) / (df[w] + 1)) + 1.0
            tfidf[w] += idf
    return tfidf


def is_meaningful_ascii(w: str, keep_set=set()) -> bool:
    return w in keep_set


def drop_bad_phrase(phrase: str, promo_ascii=set(), drop_regex=[], keep_ascii=set()) -> bool:
    low = phrase.lower()
    if re.fullmatch(r"[a-z0-9 .]+", low) and not is_meaningful_ascii(low.strip(), keep_ascii):
        return True
    if any(tok in promo_ascii for tok in low.split()):
        return True
    for rgx in drop_regex:
        if rgx.search(low):
            return True
    return False

    return sorted_words[:top_n]


def extract_phrases(posts, top_n=15, stopwords=None, promo_ascii=None, drop_regex=None, keep_ascii=None, ignore_tokens=None, pos_filters=None):
    """
    Extracts phrases (bigrams/trigrams) from posts using PMI.
    Now supports POS filtering to ensure phrases are made of meaningful words.
    """
    if not JIEBA_OK:
        return []

    if stopwords is None: stopwords = set()
    if promo_ascii is None: promo_ascii = set()
    if drop_regex is None: drop_regex = []
    if keep_ascii is None: keep_ascii = set()
    if ignore_tokens is None: ignore_tokens = set()
    if pos_filters is None: pos_filters = []

    unigram = Counter()
    bigram = Counter()
    trigram = Counter()

    for p in posts:
        # Use pseg for POS tagging if pos_filters are active, otherwise fallback to cut
        # To be consistent, let's use pseg if available
        words = pseg.cut(p)
        
        valid_post_tokens = []
        for w, flag in words:
            w = w.strip().lower()
            if not w: continue
            
            # --- SAME FILTERING LOGIC AS KEYWORDS ---
            if len(w) < 2 and w not in ["ai", "ui", "ux"] and not re.match(r'^[a-zA-Z0-9]+$', w): 
                continue
            if w in stopwords or w in ignore_tokens:
                continue
            
            if pos_filters:
                matched_pos = False
                for pf in pos_filters:
                    if flag.startswith(pf):
                        matched_pos = True
                        break
                if not matched_pos:
                    continue
            # ----------------------------------------
            
            valid_post_tokens.append(w)

        toks = valid_post_tokens
        unigram.update(toks)
        bigram.update(zip(toks, toks[1:]))
        trigram.update(zip(toks, toks[1:], toks[2:]))

    total_unigrams = sum(unigram.values()) or 1

    def pmi2(a, b):
        # Additional check: prevent repeating same word "劇透 劇透"
        if a == b: return -999.0
        
        pab = bigram[(a, b)] / total_unigrams
        pa = unigram[a] / total_unigrams
        pb = unigram[b] / total_unigrams
        if pab == 0 or pa == 0 or pb == 0:
            return -999.0
        return math.log((pab / (pa * pb)) + 1e-12)

    def pmi3(a, b, c):
        if a == b or b == c or a == c: return -999.0
        return 0.5 * (pmi2(a, b) + pmi2(b, c))

    phrase_scores = []
    phrase_min_freq = 3 # Hardcoded
    phrase_pmi_min = 3.0 # Hardcoded

    for (a, b), f in bigram.items():
        if f >= phrase_min_freq:
            s = pmi2(a, b)
            cand = f"{a} {b}"
            if s >= phrase_pmi_min and not drop_bad_phrase(cand, promo_ascii, drop_regex, keep_ascii):
                phrase_scores.append((cand, f, s))
    for (a, b, c), f in trigram.items():
        if f >= phrase_min_freq:
            s = pmi3(a, b, c)
            cand = f"{a} {b} {c}"
            if s >= phrase_pmi_min and not drop_bad_phrase(cand, promo_ascii, drop_regex, keep_ascii):
                phrase_scores.append((cand, f, s))

    phrase_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return phrase_scores[:top_n]


def analyze_file(path, topn=30, stopwords_path=None, min_len=2,
                 keep_hash_at=False, pos_only=False,
                 top_phrases=15, phrase_min_freq=3, phrase_pmi_min=3.0,
                 out_csv="word_tfidf.csv", out_txt="word_tfidf.txt",
                 out_phrase_csv="phrase_freq.csv", out_hashtag_csv="hashtag_freq.csv",
                 debug=False, user_dict_path=None, min_doc_tokens=3, min_chinese_ratio=0.2,
                 precision=False, progress=True, dedupe=True, dedupe_hamming=3,
                 config_path=None, default_stopwords_path=None,
                 engine="jieba", ckip_level=1, device=-1):
    if not os.path.exists(path):
        raise FileNotFoundError(f"input file not found: {path}")

    if JIEBA_OK:
        # 自動載入台灣流行語字典 (如果存在)
        tw_slang_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config", "tw_slang.txt")
        if not os.path.exists(tw_slang_path):
            tw_slang_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "tw_slang.txt")
        if os.path.exists(tw_slang_path):
            try:
                jieba.load_userdict(tw_slang_path)
            except Exception:
                pass
        # 載入使用者自訂字典
        if user_dict_path and os.path.exists(user_dict_path):
            try:
                jieba.load_userdict(user_dict_path)
            except Exception:
                pass

    # Load Config
    config = load_analysis_config(config_path) if config_path else {}
    
    drop_patterns = config.get("drop_patterns", [])
    
    promo_ascii = set(config.get("promo_ascii", []))
    phrase_drop_regex = config.get("phrase_drop_regex", [])
    
    meaningful_ascii = set(config.get("meaningful_ascii_keep", []))
    noisy_ascii = set(config.get("noisy_ascii", []))
    precision_stopwords = set(config.get("precision_stopwords", []))

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = clean_text(raw)
    posts = split_posts(raw, drop_patterns=drop_patterns)
    # Strip Threads UI artefacts (time prefix, engagement numbers, etc.)
    posts = [strip_threads_metadata(p) for p in posts]
    posts = [p for p in posts if p]
    stopwords = load_stopwords(stopwords_path, default_stopwords_path, noisy_ascii)

    if precision:
        # precision 模式：更嚴格的中文比例、每篇至少 token 數、二值 TF、預設用詞性
        if not pos_only:
            pos_only = True
        min_chinese_ratio = max(min_chinese_ratio, 0.4)
        min_doc_tokens = max(min_doc_tokens, 5)
        # 口水詞再擴充
        stopwords |= precision_stopwords

    def chinese_ratio(s: str) -> float:
        if not s:
            return 0.0
        ch = len(re.findall(r"[\u4e00-\u9fff]", s))
        return ch / max(1, len(s))

    filtered_posts = []
    for p in iter_with_progress(posts, label="filtering", enable=progress):
        if chinese_ratio(p) >= min_chinese_ratio:
            filtered_posts.append(p)

    if debug:
        print(f"[DEBUG] posts total: {len(posts)}, after zh-ratio filter: {len(filtered_posts)}")
        for i, p in enumerate(filtered_posts[:5]):
            print(f"[DEBUG] sample post {i}: {p[:120].replace('\n',' ')}...")

    docs_tokens = []
    hashtags = Counter()
    seen_norm = set()
    simhash_buckets = {}
    dup_exact = 0
    dup_near = 0

    # --- ENGINE-SPECIFIC TOKENIZATION ---
    ckip_results = None
    if engine.lower() == "ckip":
        if not CKIP_OK:
            print("[WARN] ckip-transformers not installed. Falling back to jieba.")
        else:
            analyzer = CKIPAnalyzer(level=ckip_level, device=device)
            # Batch process filtered posts
            ckip_results = analyzer.analyze(filtered_posts)
            
            # Fill docs_tokens for compatible processing (tfidf, etc.)
            for i, (ws, pos, ner) in enumerate(ckip_results):
                toks = []
                for w, flag in zip(ws, pos):
                    w = w.strip().lower()
                    if len(w) >= min_len and w not in stopwords:
                        toks.append(w)
                docs_tokens.append(toks)
                # Hashtag extraction
                for tag in re.findall(r"#[\w\u4e00-\u9fff]+", filtered_posts[i]):
                    hashtags[tag.lower()] += 1

    if not ckip_results:
        # Fallback/Default Jieba behavior
        for p in iter_with_progress(filtered_posts, label="tokenizing", enable=progress):
            norm = normalize_text_for_dedupe(p)
            if norm in seen_norm:
                dup_exact += 1
                continue
            seen_norm.add(norm)
            
            toks = tokenize_post(p, keep_hash_at=keep_hash_at, pos_only=pos_only, stopwords=stopwords, min_len=min_len)
            # 額外過濾垃圾
            toks = [t for t in toks if t not in noisy_ascii and not re.fullmatch(r"[a-z]{2,3}\d*", t)]
            if keep_hash_at:
                toks = [t for t in toks if not looks_like_username(t) or t.startswith("@")]
            else:
                toks = [t for t in toks if not looks_like_username(t)]
            
            if len(toks) >= min_doc_tokens:
                if dedupe:
                    sig = simhash(toks)
                    if is_near_duplicate(sig, simhash_buckets, threshold=dedupe_hamming):
                        dup_near += 1
                        continue
                docs_tokens.append(toks)
                for tag in re.findall(r"#[\w\u4e00-\u9fff]+", p):
                    hashtags[tag.lower()] += 1

    if debug:
        total_tokens = sum(len(t) for t in docs_tokens)
        non_empty_docs = sum(1 for t in docs_tokens if t)
        print(f"[DEBUG] non-empty docs: {non_empty_docs}/{len(filtered_posts)}, total tokens: {total_tokens}")
        if dedupe:
            print(f"[DEBUG] dedupe exact: {dup_exact}, near: {dup_near}")
        if non_empty_docs <= 2 or total_tokens < 50:
            print("[WARN] Very few tokens left. Consider loosening filters or stopwords.")

    # ---- 單詞：TF-IDF 或 Binary TF-IDF ----
    if docs_tokens:
        tfidf = compute_tfidf_binary(docs_tokens) if precision else compute_tfidf(docs_tokens)
        top_words = tfidf.most_common(topn)
    else:
        top_words = []

    # 4. Keyword Analysis (TF-IDF with POS filtering)
    top_keywords = extract_keywords_with_pos(
        filtered_posts, 
        top_n=topn, 
        stopwords=stopwords,
        ignore_tokens=config.get("ignore_tokens"),
        pos_filters=list(config.get("pos_filters", [])),
        ckip_results=ckip_results
    )
    


    # 5. Phrase Analysis (Bigrams / Trigrams)
    # We pass the same ignore_tokens to phrase extraction if possible, 
    # but phrase extraction uses 'cut' internally. 
    # Ideally we refactor 'extract_phrases' too, but for now let's just use the filtered keywords to guide judgment?
    # Or just let it run. The 'drop_patterns' in split_posts already removed repetitive lines.
    top_phr = extract_phrases(
        posts, 
        top_n=top_phrases, 
        stopwords=stopwords,
        promo_ascii=config.get("promo_ascii", []),
        drop_regex=config.get("phrase_drop_regex", []),
        keep_ascii=config.get("meaningful_ascii_keep", set()),
        ignore_tokens=config.get("ignore_tokens", set()),
        pos_filters=list(config.get("pos_filters", []))
    )

    # ---- 儲存結果 ----
    with open(out_txt, "w", encoding="utf-8") as f:
        for w, s in top_keywords:
            f.write(f"{w},{s:.4f}\n")

    if pd:
        pd.DataFrame([(w, s) for w, s in top_keywords], columns=["word","tfidf_score"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
        if top_phr:
            pd.DataFrame(top_phr, columns=["phrase","freq","approx_pmi"]).to_csv(out_phrase_csv, index=False, encoding="utf-8-sig")
        if hashtags:
            pd.DataFrame(sorted(hashtags.items(), key=lambda x: x[1], reverse=True), columns=["hashtag","count"]).to_csv(out_hashtag_csv, index=False, encoding="utf-8-sig")
    else:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("word,tfidf_score\n")
            for w, s in top_keywords:
                f.write(f"{w},{s:.4f}\n")
        if top_phr:
            with open(out_phrase_csv, "w", encoding="utf-8") as f:
                f.write("phrase,freq,approx_pmi\n")
                for p, fr, sc in top_phr:
                    f.write(f"{p},{fr},{sc:.4f}\n")
        if hashtags:
            with open(out_hashtag_csv, "w", encoding="utf-8") as f:
                f.write("hashtag,count\n")
                for h, c in sorted(hashtags.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{h},{c}\n")

    # ---- 趨勢觀察簡報 (Trend Report) ----
    generate_trend_report(top_keywords, top_phr, hashtags, ckip_results=ckip_results)
    
    if debug:
        print(f"\n[INFO] Analysis complete. Saved: {out_csv}, {out_phrase_csv}, {out_hashtag_csv}, and {out_txt}")
    return top_keywords, top_phr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze hot keywords from Threads crawl with TF-IDF & phrase mining (precision mode supported).")
    ap.add_argument("--input", "-i", default="result.txt", help="input text file (default: result.txt)")
    ap.add_argument("--top", "-n", type=int, default=30, help="top N words to output")
    ap.add_argument("--stopwords", "-s", default=None, help="optional stopwords file (one per line)")
    ap.add_argument("--minlen", type=int, default=2, help="minimum token length to keep (default 2)")
    ap.add_argument("--keep-hash-at", action="store_true", help="keep @user and #tag tokens")
    ap.add_argument("--pos-only", action="store_true", help="keep nouns-like POS only (jieba pos tagging)")
    ap.add_argument("--top-phrases", type=int, default=15, help="how many phrases to show")
    ap.add_argument("--phrase-min-freq", type=int, default=3, help="minimum frequency of a phrase candidate")
    ap.add_argument("--phrase-pmi-min", type=float, default=3.0, help="minimum PMI for a phrase")
    ap.add_argument("--debug", action="store_true", help="print debug info")
    ap.add_argument("--user-dict", default=None, help="jieba user dict path (optional)")
    ap.add_argument("--min-doc-tokens", type=int, default=3, help="min tokens per post to keep")
    ap.add_argument("--min-chinese-ratio", type=float, default=0.2, help="min Chinese char ratio per post [0..1]")
    ap.add_argument("--precision", action="store_true",
                    help="stricter filters + binary TF-IDF + nouns only")
    ap.add_argument("--engine", default="jieba", choices=["jieba", "ckip"], help="NLP engine (default: jieba)")
    ap.add_argument("--ckip-level", type=int, default=1, choices=[1, 2, 3], help="CKIP model level (1:tiny, 2:base, 3:bert)")
    ap.add_argument("--device", default="cpu", help="device for CKIP (cpu, cuda, or device index)")
    ap.add_argument("--line-posts", action="store_true", default=None,
                    help="treat each line as a post (auto-detect when omitted)")
    ap.add_argument("--no-dedupe", action="store_true", help="disable near-duplicate removal")
    ap.add_argument("--dedupe-hamming", type=int, default=3, help="max Hamming distance for simhash dedupe")
    ap.add_argument("--no-progress", action="store_true", help="disable progress bar")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Establish default config paths
    config_dir = os.path.join(script_dir, "config")
    config_yaml = os.path.join(config_dir, "analysis_config.yaml")
    stopwords_default = os.path.join(config_dir, "stopwords.txt")

    local_result = os.path.join(script_dir, "result.txt")
    web_spider_result = os.path.join(script_dir, "web-spider", "result.txt")
    
    if args.input != "result.txt":
        input_path = args.input
    elif os.path.exists(local_result):
        input_path = local_result
    elif os.path.exists(web_spider_result):
        input_path = web_spider_result
    else:
        input_path = args.input

    # Choose device index
    device_val = -1
    if args.device == "cuda":
        device_val = 0
    elif args.device.isdigit():
        device_val = int(args.device)

    # Create output directory if needed
    os.makedirs("outputs", exist_ok=True)
    
    analyze_file(
        input_path,
        topn=args.top,
        stopwords_path=args.stopwords,
        min_len=args.minlen,
        keep_hash_at=args.keep_hash_at,
        pos_only=args.pos_only,
        top_phrases=args.top_phrases,
        phrase_min_freq=args.phrase_min_freq,
        phrase_pmi_min=args.phrase_pmi_min,
        out_csv="outputs/word_tfidf.csv",
        out_txt="outputs/word_tfidf.txt",
        out_phrase_csv="outputs/phrase_freq.csv",
        out_hashtag_csv="outputs/hashtag_freq.csv",
        debug=args.debug,
        user_dict_path=args.user_dict,
        min_doc_tokens=args.min_doc_tokens,
        min_chinese_ratio=args.min_chinese_ratio,
        precision=args.precision,
        progress=not args.no_progress,
        dedupe=not args.no_dedupe,
        dedupe_hamming=args.dedupe_hamming,
        config_path=config_yaml,
        default_stopwords_path=stopwords_default,
        engine=args.engine,
        ckip_level=args.ckip_level,
        device=device_val,
    )
