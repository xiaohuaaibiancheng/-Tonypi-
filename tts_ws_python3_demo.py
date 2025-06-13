# xfyun_tts.py
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os
import wave

class XFYunTTS:
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret

    def create_valid_wav(self, pcm_data, sample_rate=16000, channels=1, sample_width=2, output_path='./demo.wav'):
        """将PCM数据转换为标准WAV格式"""
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

    def on_message(self, ws, message):
        try:
            message = json.loads(message)
            code = message["code"]
            sid = message["sid"]
            audio = message["data"]["audio"]
            audio = base64.b64decode(audio)
            status = message["data"]["status"]

            if status == 2:
                print("合成完成，正在生成WAV文件...")
                # 读取临时PCM文件
                with open('./temp.pcm', 'rb') as pcm_file:
                    pcm_data = pcm_file.read()
                # 转换为WAV
                self.create_valid_wav(pcm_data, output_path=self.output_path)
                # 清理临时文件
                os.remove('./temp.pcm')
                print(f"WAV文件已生成: {self.output_path}")
                ws.close()

            if code != 0:
                print(f"错误: {message['message']} (code: {code})")
            else:
                # 先写入临时PCM文件
                with open('./temp.pcm', 'ab') as f:
                    f.write(audio)

        except Exception as e:
            print("处理音频数据异常:", e)

    def on_error(self, ws, error):
        print("### 错误:", error)

    def on_close(self, ws, close_status_code, close_msg):
        print("### 连接关闭 ###")
        print(f"状态码: {close_status_code}")
        print(f"消息: {close_msg}")

    def on_open(self, ws):
        def run(*args):
            data = {
                "common": self.CommonArgs,
                "business": self.BusinessArgs,
                "data": self.Data
            }
            ws.send(json.dumps(data))
            # 清理旧文件
            for f in ['./temp.pcm', self.output_path]:
                if os.path.exists(f):
                    os.remove(f)

        thread.start_new_thread(run, ())

    def text_to_speech(self, text, output_path='./demo.wav'):
        self.output_path = output_path
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {
            "aue": "raw",  # 使用原始PCM格式
            "auf": "audio/L16;rate=16000",  # 16kHz, 16-bit
            "vcn": "x4_xiaoyan",  # 使用标准发音人
            "tte": "utf8"
        }
        self.Data = {
            "status": 2,
            "text": str(base64.b64encode(text.encode('utf-8')), "UTF8")
        }

        def create_url():
            url = 'wss://tts-api.xfyun.cn/v2/tts'
            now = datetime.now()
            date = format_date_time(mktime(now.timetuple()))

            signature_origin = "host: ws-api.xfyun.cn\ndate: " + date + "\nGET /v2/tts HTTP/1.1"
            signature_sha = hmac.new(
                self.APISecret.encode('utf-8'),
                signature_origin.encode('utf-8'),
                digestmod=hashlib.sha256
            ).digest()
            signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

            authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
            authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

            v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
            return url + '?' + urlencode(v)

        websocket.enableTrace(False)
        wsUrl = create_url()
        print("WebSocket连接URL:", wsUrl)

        ws = websocket.WebSocketApp(
            wsUrl,
            on_message=lambda ws, msg: self.on_message(ws, msg),
            on_error=lambda ws, err: self.on_error(ws, err),
            on_close=lambda ws, code, msg: self.on_close(ws, code, msg),
            on_open=lambda ws: self.on_open(ws)
        )
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})