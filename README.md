# Threads 流行趨勢分析工具 (v3.5 Developer Edition)

本工具是一套全自動化的 Threads 趨勢洞察系統。從數據抓取、自然語言處理 (NLP)、AI 摘要，到視覺化圖表產出與 LINE 自動通報，只需簡單幾行指令即可完成。

---

## ✨ 核心功能
- **🤖 AI 智慧摘要**：整合 Google Gemini API，自動產出今日話題懶人包與策略建議。
- **🧠 高精準 NLP 引擎**：預設使用中研院 CKIP 實體辨識 (NER)，精準分類人物、作品、地點。
- **📊 數據視覺化**：自動產出熱門趨勢長條圖與視覺化詞雲圖 (Word Cloud)。
- **📱 LINE 自動通報**：直接將分析結果推送到手機，適合行動辦公。

---

## 🚀 快速開始 (四步驟)

### 1. 安裝與設定
```bash
# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
# 請根據 .env.example 建立 .env 檔案並填入：
# GEMINI_API_KEY
# LINE_CHANNEL_ACCESS_TOKEN
# LINE_USER_ID
```

### 2. 資料抓取
```bash
# 啟動爬蟲 (會開啟瀏覽器供登入，登入後請在終端機按 Enter 繼續)
python web-spider/src/crawler.py
```

### 3. 一鍵分析
```bash
# 全功能模式：AI 摘要 + 繪圖 + 傳送到 LINE
python analyze.py --line
```

### 4. 查看成果
- **終端機**：即時查看排版精美的「Premium 報表」。
- **輸出資料夾**：
  - `outputs/word_tfidf.csv` (結構化數據)
  - `outputs/visuals/*.png` (視覺化圖表)
- **手機**：檢查您的 LINE 通知！

---

## 🛠️ 進階指令參考
- **切換 Jieba 引擎 (極速)**：`python analyze.py --engine jieba`
- **使用 GPU 加速 (CKIP)**：`python analyze.py --device cuda`
- **手動產出圖表**：`python visualize.py`

---

## 🏆 總結
本工具旨在為社群經營者與分析師提供最直觀、最純淨的數據洞察。所有的雜訊過濾與排版優化皆已模組化，讓您只需專注於數據背後的價值。
