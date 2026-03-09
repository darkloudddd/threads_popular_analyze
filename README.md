# Threads 熱門話題分析工具 (v2.0)

## 簡介
此工具專為個人分析使用設計，包含兩個核心組件：
1. **爬蟲 (Web Spider)**：使用 Playwright 模擬瀏覽器，支援手動登入以繞過驗證，自動捲動抓取 Threads 貼文。
2. **分析器 (Analyzer)**：支援 **Jieba** 與 **中研院 CKIP** 雙引擎分析。具備 TF-IDF 權重、SimHash 去重與 NER (實體辨識) 功能，能自動產出分類趨勢報告。

## 快速開始

### 1. 安裝環境
確保已安裝 Python 3.10 或以上版本（CKIP 建議 3.10+）。
```bash
# 安裝所有必要與進階requirements
pip install -r requirements.txt
```

### 2. 執行爬蟲
首次執行需要手動登入：
```bash
python web-spider/src/main.py --user-data "user_data" --output "result.txt" --max-scrolls 200
```
- **步驟**：登入後回到終端機按下 **Enter** 即可開始自動抓取。

### 3. 執行分析
抓取完成後（預設讀取 `result.txt`），執行分析腳本：

#### 模式 A：極速分析 (Jieba)
```bash
python analyze.py --precision
```

#### 模式 B：高精準分析 (中研院 CKIP)
具備實體辨識 (NER)，能精準抓出人名、作品與地點。
```bash
python analyze.py --engine ckip --ckip-level 1
```

- **參數說明**：
    - `--precision`: 開啟精準模式，套用嚴格雜訊過濾並優化詞權重。
    - `--engine`: 選擇分析引擎 `jieba` (預設) 或 `ckip`。
    - `--device`: CKIP 可選 `cpu` (預設) 或 `cuda` (GPU 加速)。
    - `--top`: 設定顯示前 N 個結果（預設 30）。
    - `--ckip-level`: CKIP 模型等級（1: Tiny 最快, 2: Base 平衡, 3: BERT 最準並含 NER）。

## 輸出結果
分析結果將自動儲存於 **`outputs/`** 資料夾：
- `outputs/word_tfidf.csv`: 關鍵字權重排行。
- `outputs/phrase_freq.csv`: 熱門詞組（如：陽光女子合唱團）。
- `outputs/hashtag_freq.csv`: 熱門 Hashtag 統計。
- **終端機報告**：自動分類「人物影視、熱門地點、美食生活、事件話題」的趨勢簡報。

## 進階設定
- `config/stopwords.txt`: **停用詞表**。
- `config/tw_slang.txt`: **台灣流行語自訂詞庫**。
- `config/analysis_config.yaml`: **分析器邏輯設定**（包含雜訊 Regex、詞性加權等）。

## 常見問題
- **Q: 執行分析時出現 Emoji 編碼錯誤？**
  - A: 程式已加入編碼保護邏輯。若仍有問題，請確保終端機支援 UTF-8，或確認 python 已更新至最新版。
- **Q: CKIP 跑很慢？**
  - A: CKIP 基於深度學習模型（Transformer），處理速度較慢為正常現象。若有 GPU，可加上 `--device cuda` 加速。
