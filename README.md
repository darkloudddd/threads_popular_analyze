# Threads 熱門話題分析工具

## 簡介
此工具專為個人分析使用設計，包含兩個核心組件：
1. **爬蟲 (Web Spider)**：使用 Playwright 模擬瀏覽器，支援手動登入以繞過驗證，自動捲動抓取 Threads 貼文。
2. **分析器 (Analyzer)**：使用 NLP 技術（TF-IDF、SimHash 去重）分析抓取內容，找出熱門關鍵字與話題。

## 快速開始

### 1. 安裝環境
確保已安裝 Python 3.8 或以上版本。
```bash
# 安裝依賴套件
pip install -r web-spider/requirements.txt
```

### 2. 執行爬蟲
首次執行需要手動登入：
```bash
python web-spider/src/main.py --user-data "user_data" --output "result.txt" --max-scrolls 200
```
- **步驟**：
    1. 程式會自動開啟一個 Chrome 瀏覽器視窗。
    2. 請在該視窗中登入您的 Threads 帳號。
    3. 登入成功後，回到終端機按下 **Enter** 鍵。
    4. 爬蟲將開始自動捲動並收集貼文。
- **參數說明**：
    - `--max-scrolls`: 捲動次數（預設 500）。
    - `--pause`: 每次捲動暫停秒數（預設 1.0）。

### 3. 執行分析
抓取完成後，執行分析腳本：
```bash
python analyze.py --input result.txt --precision --top 50
```
- **參數說明**：
    - `--precision`: 開啟精準模式（建議使用），會套用更嚴格的雜訊過濾。
    - `--top`: 顯示前 N 個熱門關鍵字。
- **輸出結果**：
    - `word_tfidf.csv`: 關鍵字權重表。
    - `phrase_freq.csv`: 熱門詞組/短語（例如 "交換禮物"、"新年快樂"）。
    - `hashtag_freq.csv`: 熱門 Hashtag 統計。

## 進階設定
為方便維護，所有過濾邏輯皆已移至設定檔，不需修改程式碼：

| 設定檔路徑 | 用途 |
|Orz|Orz|
| `config/stopwords.txt` | **停用詞表**：一行一個，可自行新增想過濾的詞（如 "覺得"、"可能"）。 |
| `config/analysis_config.yaml` | **分析設定**：定義雜訊 Regex、詞性過濾 (POS Filter)、忽略詞清單等。 |
| `web-spider/src/config/config.yaml` | **爬蟲設定**：定義 CSS 選擇器、目標網址、視窗大小。 |

## 常見問題
- **Q: 爬蟲跑一跑停住了？**
  - A: 請檢查瀏覽器視窗是否被關閉，或網路是否不穩。可嘗試增加 `--pause` 時間。
- **Q: 分析結果很多雜訊？**
  - A: 請編輯 `config/stopwords.txt` 將雜訊詞加入，或檢查 `config/analysis_config.yaml` 中的過濾規則。
