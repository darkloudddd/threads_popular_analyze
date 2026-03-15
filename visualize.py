import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.font_manager as fm

# --- 1. 中文字型設定 (Windows 優先) ---
def setup_chinese_font():
    # 常用中文字型名稱列表
    chinese_fonts = ['Microsoft JhengHei', 'DFKai-SB', 'SimHei', 'Noto Sans TC']
    for font_name in chinese_fonts:
        # 檢查字型是否安裝
        if any(f.name == font_name for f in fm.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
            print(f"[*] 使用字型: {font_name}")
            return font_name
    print("[!] 找不到合適的中文字型，圖表可能會出現亂碼。")
    return None

# --- 2. 繪製長條圖 ---
def plot_bar_chart(df, x_col, y_col, title, output_path, color='skyblue'):
    plt.figure(figsize=(10, 6))
    # 限制前 15 筆
    plot_df = df.head(15).copy()
    
    sns.barplot(data=plot_df, x=y_col, y=x_col, palette="viridis" if color=='viridis' else None, color=color if color!='viridis' else None)
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('權重 / 頻率', fontsize=12)
    plt.ylabel('項目', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] 圖表已儲存至: {output_path}")

# --- 3. 生成詞雲圖 ---
def generate_wordcloud(freq_dict, title, output_path, font_path=None):
    if not freq_dict:
        return
        
    wc = WordCloud(
        font_path=font_path, 
        width=1200, 
        height=800, 
        background_color='white',
        colormap='tab10',
        max_words=100
    ).generate_from_frequencies(freq_dict)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] 詞雲圖已儲存至: {output_path}")

# --- 4. 繪製分類趨勢圖 ---
def plot_categorized_trends(cat_data, output_path):
    """Draw a horizontal bar chart showing word counts per category."""
    categories = []
    counts = []
    
    for cat, items in cat_data.items():
        if items:
            categories.append(cat)
            counts.append(len(items))
            
    if not categories:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=categories, palette="magma")
    plt.title('各項板塊話題熱度分布', fontsize=16, pad=20)
    plt.xlabel('熱門詞彙數量', fontsize=12)
    plt.ylabel('趨勢分類', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] 分類趨勢圖已儲存至: {output_path}")

# --- 主程式 ---
def run_visualization():
    setup_chinese_font()
    
    output_dir = "outputs"
    visuals_dir = os.path.join(output_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    
    # 優先處理分類數據 (Architecture Realignment)
    import json
    cat_file = os.path.join(output_dir, "categorized_trends.json")
    if os.path.exists(cat_file):
        try:
            with open(cat_file, "r", encoding="utf-8") as f:
                cat_data = json.load(f)
                plot_categorized_trends(cat_data, os.path.join(visuals_dir, "category_distribution.png"))
        except Exception as e:
            print(f"[!] 無法讀取分類 JSON: {e}")

    # --- 關鍵字分析 ---
    word_file = os.path.join(output_dir, "word_tfidf.csv")
    if os.path.exists(word_file):
        df_word = pd.read_csv(word_file)
        if not df_word.empty:
            # 長條圖
            plot_bar_chart(df_word, 'word', 'weighted_score', '熱門話題關鍵字 Top 15', 
                           os.path.join(visuals_dir, "top_keywords.png"))
            
            # 詞雲圖
            # 獲取字型路徑供 WordCloud 使用
            current_font = plt.rcParams['font.sans-serif'][0]
            font_prop = fm.FontProperties(family=current_font)
            font_path = fm.findfont(font_prop)
            
            # 轉字典
            word_freq = dict(zip(df_word['word'], df_word['weighted_score']))
            generate_wordcloud(word_freq, 'Threads 話題詞雲', 
                               os.path.join(visuals_dir, "wordcloud.png"), 
                               font_path=font_path)

    # --- Hashtag 分析 ---
    hashtag_file = os.path.join(output_dir, "hashtag_freq.csv")
    if os.path.exists(hashtag_file):
        df_hash = pd.read_csv(hashtag_file)
        if not df_hash.empty:
            plot_bar_chart(df_hash, 'hashtag', 'weighted_score', '最受歡迎 Hashtag Top 10', 
                           os.path.join(visuals_dir, "top_hashtags.png"), color='lightcoral')

if __name__ == "__main__":
    run_visualization()
