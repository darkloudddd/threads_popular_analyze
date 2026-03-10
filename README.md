# 🧶 Threads 爆紅趨勢分析工具 v3.1 

想知道今天 Threads 上大家都在聊什麼？或是哪位公眾人物又引起話題了嗎？這個工具可以幫您自動抓取貼文、計算熱度，並用 **AI 幫您總結懶人包**！

---

## 🚀 三分鐘快速上手

請跟著以下步驟操作：

### 第一步：安裝環境 (只需做一次)
打開您的終端機 (Terminal/CMD)，輸入：
```bash
# 1. 下載必要的套件
pip install -r requirements.txt

# 2. 安裝瀏覽器驅動（這是爬蟲需要的「動力」）
playwright install chromium
```

### 第二步：設定 AI 助手 (選配功能，建議開啟)
1. 找到資料夾中的 `.env.example` 檔案，把它改名為 **`.env`**。
2. 至 [Google AI Studio](https://aistudio.google.com/) 免費申請一串 API Key。
3. 把變更後的 `.env` 打開，貼上您的 Key：`GEMINI_API_KEY=您的代碼`。

### 第三步：開始抓取資料 (自動爬蟲)
執行以下指令，程式會自動開啟瀏覽器 (後面數字是 scroll 的次數)：
```bash
python web-spider/src/main.py --max-scrolls 100
```
> **💡 小提醒**：程式開啟後會**先暫停**。您可以手動登入、關閉彈窗，**確認頁面就緒後**，回到終端機畫面按下 **Enter** 鍵即可開始自動採集！

### 第四步：產生趨勢簡報 (AI 加權分析)
資料抓完後，輸入這行指令讓 AI 幫您分析：
```bash
python analyze.py
```
> **✨ 效果**：程式會**自動啟動 AI 摘要**。它會產出 CSV 報表，並在螢幕上直接印出 **「今日話題懶人包」** 與社群策略建議！

---

## 🛠️ 常見指令彙整

如果您想進行更進階的操作：

| 目標 | 指令 |
| :--- | :--- |
| **只想爬某個人的貼文** | `python web-spider/src/main.py --urls "人家的Threads網址"` |
| **關閉 AI 摘要 (節省 API 額度)** | `python analyze.py --no-ai` |
| **關閉精準模式 (顯示原始數據)** | `python analyze.py --no-precision` |
| **切換回極速引擎 (Jieba)** | `python analyze.py --engine jieba` |
| **顯示更多結果 (預設為30)** | `python analyze.py --top 50` |

## 📁 產出檔案在哪裡？
分析完成後，請到 **`outputs/`** 資料夾查看：
- `word_tfidf.csv`: 最熱門的關鍵字排行榜。
- `phrase_freq.csv`: 大家的常用詞組。
- `hashtag_freq.csv`: 最火熱的標籤數量。

## ❓ 常見問題 Q&A
- **Q: 為什麼 AI 摘要沒出現？**
  - A: 請確認 `.env` 檔案名稱正確，且 Key 沒有填錯。
- **Q: 爬蟲卡住了怎麼辦？**
  - A: 如果畫面長時間沒動，可以按 `Ctrl + C` 強制停止，或是檢查是否需要重新登入。
