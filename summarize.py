import os
import time
import asyncio
import warnings

# 隱藏不必要的警告 (如 Google Generative AI 的已過時提示)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*All support for the `google.generativeai` package has ended.*")

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class Summarizer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        # Start with a stable alias
        self.model_name = 'gemini-flash-latest' 
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None

    def is_available(self):
        return self.model is not None

    async def _generate_with_retry(self, prompt, max_retries=3):
        """
        Helper to generate content with exponential backoff for 429 errors 
        and automatic model fallback for 404/Quota issues.
        """
        if not self.api_key:
            return "AI Summary unavailable (No API Key)."

        # Order of preference for models
        models_to_try = [
            'gemini-flash-latest', 
            'gemini-1.5-flash', 
            'gemini-2.0-flash', 
            'gemini-pro-latest'
        ]
        
        last_error = ""

        for model_id in models_to_try:
            try:
                temp_model = genai.GenerativeModel(model_id)
                for i in range(max_retries):
                    try:
                        # Attempt generation
                        # Note: We use the block execution since we are in an async wrapper
                        response = temp_model.generate_content(prompt)
                        if response and response.text:
                            return response.text
                        else:
                            return "AI generated an empty response."
                    except Exception as e:
                        err_str = str(e)
                        # Handle Rate Limit / Quota
                        if "429" in err_str:
                            if "limit: 0" in err_str:
                                # This model is completely restricted for this key, move to next model
                                print(f"[*] Model {model_id} quota 0. Trying next model...")
                                break
                            
                            wait_time = (i + 1) * 3
                            print(f"[*] API Rate Limit (429) hit for {model_id}. Retrying in {wait_time}s... (Attempt {i+1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        # Handle Model Not Found
                        if "404" in err_str:
                            # print(f"[*] Model {model_id} not found. Trying next...")
                            break
                            
                        # Other errors (e.g. Safety Filters)
                        return f"AI Generation stopped: {err_str}"
                
            except Exception as e:
                last_error = str(e)
                continue
        
        return f"Error: All Gemini models failed or Quota exceeded. Details: {last_error}"

    async def generate_summary(self, top_posts):
        """
        Generates a summary of the top trending posts.
        """
        if not self.is_available():
            return "AI Summary unavailable (No API Key)."

        # Prepare prompt (Use clean_text to avoid noise like 1/2)
        p_text = "\n----- \n".join([f"Post: {p.get('clean_text', p['text'])}\nEngagement: {p.get('likes',0)} likes" for p in top_posts[:15]])
        
        prompt = f"""
        你是一位社群趨勢專家。以下是從 Threads 抓取到的熱門貼文：
        {p_text}
        
        請根據以上內容，提供以下洞察：
        1. 今日話題懶人包：用 3-5 點總結今天社群在討論什麼。
        2. 社群氛圍分析：判斷整體社群氛圍。
        3. 策略建議：如果要在這波趨勢中發文，建議什麼主題或方向？
        
        【重要規範】：
        - 請用「繁體中文」回答。
        - 🚫 禁止使用任何 Markdown 格式（例如：不要用 **粗體**、不要用 ### 標題、不要用 * 列表符號）。
        - 列表請單純使用數字「1. 」「2. 」開頭。
        - 內容要極簡、專業、乾淨，適合在手機簡訊中直接閱讀。
        """

        return await self._generate_with_retry(prompt)

    async def get_sentiment_tag(self, text):
        """
        Gets a single sentiment tag for a piece of text.
        """
        if not self.is_available(): return "Neutral"
        
        prompt = f"請對這段文字進行情緒標記（僅輸出一個詞，例如：炎上、好事、幽默、日常、爭議）：\n{text}"
        res = await self._generate_with_retry(prompt, max_retries=1)
        if "Error" in res:
            return "Unknown"
        return res.strip()[:10]
