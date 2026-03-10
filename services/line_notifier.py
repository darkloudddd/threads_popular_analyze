import os
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    TextMessage,
    ImageMessage,
    PushMessageRequest
)
from dotenv import load_dotenv

load_dotenv()

class LineNotifier:
    def __init__(self):
        self.access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        self.user_base_id = os.getenv("LINE_USER_ID")  # The destination user ID
        
        if self.access_token and self.user_base_id:
            self.configuration = Configuration(host="https://api.line.me", access_token=self.access_token)
            print("[*] LINE Notifier initialized.")
        else:
            self.configuration = None
            print("[!] LINE Notifier: Missing Token or User ID. Notifications disabled.")

    def is_available(self):
        return self.configuration is not None

    def send_text(self, text):
        if not self.is_available(): return
        
        with ApiClient(self.configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            push_message_request = PushMessageRequest(
                to=self.user_base_id,
                messages=[TextMessage(text=text)]
            )
            try:
                line_bot_api.push_message(push_message_request)
                print("[+] LINE: Text message sent.")
            except Exception as e:
                print(f"[!] LINE: Error sending text: {e}")

    def send_image(self, image_url, preview_url=None):
        """
        Sends an image. Requires a publicly accessible URL for LINE servers.
        """
        if not self.is_available(): return
        if not preview_url: preview_url = image_url

        with ApiClient(self.configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            push_message_request = PushMessageRequest(
                to=self.user_base_id,
                messages=[ImageMessage(originalContentUrl=image_url, previewImageUrl=preview_url)]
            )
            try:
                line_bot_api.push_message(push_message_request)
                print("[+] LINE: Image message sent.")
            except Exception as e:
                print(f"[!] LINE: Error sending image: {e}")

if __name__ == "__main__":
    # Quick Test
    notifier = LineNotifier()
    if notifier.is_available():
        notifier.send_text("Threads 分析器測試：這是一則來自 Python 的測試訊息。")
