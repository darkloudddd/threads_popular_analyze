import re
import string
import os

# --- Regex patterns for text processing ---
USERNAME_TOKEN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._]{1,28}[a-z0-9])?$", re.IGNORECASE)

# POS 保留：名詞/專有名詞/地名/機構名/動名詞等
POS_KEEP_PREFIX = ("n", "nr", "ns", "nt", "nz", "vn")
# POS 加權：專有名詞、地名、機構名加權，提升趨勢感
POS_WEIGHTS = {
    "nz": 1.5,  # 其他專名
    "nt": 1.5,  # 機構名
    "ns": 1.3,  # 地名
    "nr": 1.2,  # 人名
}

# --- 注音文 / 台灣口語轉換表 ---
ZHUYIN_MAP = {
    "ㄉ": "的", "ㄅ": "吧", "ㄇ": "嗎", "ㄋ": "呢",
    "ㄌ": "了", "ㄛ": "喔", "ㄟ": "欸", "ㄏ": "呵",
    "ㄆ": "噗", "ㄎ": "可",
}
_ZHUYIN_RE = re.compile("(" + "|".join(re.escape(k) for k in ZHUYIN_MAP) + ")")

# --- Pre-compiled Regular Expressions for Performance ---
RE_CLEAN_HTTP = re.compile(r"http\S+")
RE_CLEAN_NON_BMP = re.compile(r"[\U00010000-\U0010ffff]")
RE_CLEAN_SPACES = re.compile(r"[ \t]+")

# For is_noise
RE_NOISE_PAGINATION = re.compile(r'^\d+/\d+$')
RE_NOISE_NUMERIC = re.compile(r'^[0-9./\-: ]+$')
RE_NOISE_NUMBER_WITH_UNIT = re.compile(r'^\d+(\.\d+)?(萬|千|百|十|%|k|m|l)$', flags=re.IGNORECASE)

# For strip_threads_metadata
RE_META_FIRST_POST = re.compile(r'^第一則串文\s*')
RE_META_TIME_PREFIX = re.compile(r'^\d+\s*(小時|分鐘|天)\s*')
RE_META_TIME_REMAINING = re.compile(r'\s*還剩\d+小時\s*$')
RE_META_SUFFIX = re.compile(r'\s*(翻譯|AI\s*資訊)\s*$')
RE_META_ENGAGEMENT = re.compile(r'\s+[\d,.]+(\s*萬)?(?:\s+[\d,.]+(?:\s*萬)?){1,5}\s*$')
RE_META_CAROUSEL_START = re.compile(r'^\d+/\d+\s+')
RE_META_CAROUSEL_END = re.compile(r'\s+\d+/\d+$')
RE_META_CAROUSEL_ALONE = re.compile(r'^\d+/\d+$', flags=re.MULTILINE)
RE_META_PRODUCT_SIZE = re.compile(r'\s*[A-Za-z]{1,3}\d+(\.\d+)?(CM)?\s*')
RE_META_SYMBOLS = re.compile(r'[❗️‼️❕]+')
RE_META_SPOILER = re.compile(r'(劇透\s*)+')

# For split_posts
RE_SPLIT_DOUBLE_NL = re.compile(r'\n\s*\n')
RE_SPLIT_REPETITIVE = re.compile(r'((\S{1,4})\s+)\1{2,}')

# For inner loop text filtering
RE_ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9]+$')

def normalize_taiwanese_text(text: str) -> str:
    """Convert common zhuyin abbreviations and normalize spaces as punctuation."""
    # 1. 注音文轉換 (ㄉ -> 的, ㄅ -> 吧, ...)
    text = _ZHUYIN_RE.sub(lambda m: ZHUYIN_MAP[m.group()], text)
    # 2. 台灣用戶常以空白取代標點 — 將連續空白轉為全形逗號幫助 jieba 斷句
    #    但只在「中文字 空白 中文字」的情境下做替換，避免破壞英文
    text = re.sub(r'(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])', '，', text)
    return text

def clean_text(text):
    text = RE_CLEAN_HTTP.sub(" ", text)
    try:
        text = RE_CLEAN_NON_BMP.sub(" ", text)
    except re.error:
        pass
    # 台灣文本正規化 (注音文轉換 + 空白轉標點)
    text = normalize_taiwanese_text(text)
    text = RE_CLEAN_SPACES.sub(" ", text)
    return text

def is_noise(w):
    """Detect if a string is noise: too short, purely numeric, or purely symbolic."""
    if not w: return True
    w = w.strip(string.punctuation + " \t")
    if len(w) < 2: return True
    # Catch pagination (1/2, 5/10)
    if RE_NOISE_PAGINATION.match(w): return True
    # Catch pure numeric or punctuation
    if RE_NOISE_NUMERIC.match(w): return True
    # Catch pure number with simple unit (e.g. 2萬, 100%, 3k)
    if RE_NOISE_NUMBER_WITH_UNIT.match(w): return True
    if re.fullmatch(r'[\W_]+', w): return True
    return False

def strip_threads_metadata(text: str) -> str:
    """
    Remove Threads UI artefacts that get captured by the crawler.
    """
    text = RE_META_FIRST_POST.sub('', text)
    text = RE_META_TIME_PREFIX.sub('', text)
    text = RE_META_TIME_REMAINING.sub('', text)
    text = RE_META_SUFFIX.sub('', text)
    text = RE_META_ENGAGEMENT.sub('', text)
    text = RE_META_CAROUSEL_START.sub('', text)
    text = RE_META_CAROUSEL_END.sub('', text)
    text = RE_META_CAROUSEL_ALONE.sub('', text)
    text = RE_META_PRODUCT_SIZE.sub(' ', text)
    text = RE_META_SYMBOLS.sub('', text)
    text = RE_META_SPOILER.sub('', text)
    text.strip()
    return text

def normalize_text_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

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
