#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/pi/TonyPi/')
import cv2
import time
import os
import threading
import base64
import numpy as np
from dashscope import MultiModalConversation
import hiwonder.TTS as TTS
import hiwonder.ASR as ASR
import hiwonder.Board as Board
from hiwonder.Board import setBusServoPulse
import hiwonder.ActionGroupControl as AGC
import hiwonder.yaml_handle as yaml_handle
import json
import re
import ast
from tts_ws_python3_demo import XFYunTTS
import requests
PI_IP = '192.168.43.212'
import subprocess

def play_welcome():
    print('机器人正在说....')
    
    try:
        # 使用subprocess确保等待播放完成
        process = subprocess.Popen(
            ['aplay', '-q', '/home/pi/large_models/Record.wav'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # 等待播放完成
        process.wait()
        time.sleep(1)
        
        # 检查返回值
        if process.returncode != 0:
            print(f"播放失败，错误码: {process.returncode}")
            print(process.stderr.read().decode())
    except Exception as e:
        print(f"播放音频时出错: {e}")
    finally:
        print('机器人说完')

tts_client = XFYunTTS(
    APPID='94b1bc02',
    APISecret='YmQ3YTM2ZWM5MzZmMjAzYTc5ZDU2YzVl',
    APIKey='79c44100757b92ff8ff949ad55e155a7'
)

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
from queue import Queue
import Transport_to_words as t2w


STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

app_id = '2ccebb6f'  # 请替换为您的appid
api_key = 'bef8c5e8a0f67a425ef0fbce97ea6f78'  # 请替换为您的api_key
api_secret = 'MzMzMTk3YjMyYTAyNzIyZGFkMjE1MDU0'  # 请替换为您的api_secret
audio_file = r'./Record.wav'  # 请替换为您的音频文件路径

class Ws_Param(object):
    # 初始化
    def __init__(self):
        self.APPID = app_id
        self.APIKey = api_key
        self.APISecret = api_secret
        self.AudioFile = audio_file

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo":1,"vad_eos":10000}

    # 生成url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url


# 收到websocket消息的处理
def on_message(ws, message):
    try:
        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:
            data = json.loads(message)["data"]["result"]["ws"]
            # print(json.loads(message))
            result = ""
            for i in data:
                for w in i["cw"]:
                    result += w["w"]
            # print("sid:%s call success!,data is:%s" % (sid, json.dumps(data, ensure_ascii=False)))
            queue.put(result)
    except Exception as e:
        print("receive msg,but parse exception:", e)



# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws,a,b):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        frameSize = 8000  # 每一帧的音频大小
        intervel = 0.04  # 发送音频间隔(单位:s)
        status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧

        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                buf = fp.read(frameSize)
                # 文件结束
                if not buf:
                    status = STATUS_LAST_FRAME
                # 第一帧处理
                # 发送第一帧音频，带business 参数
                # appid 必须带上，只需第一帧发送
                if status == STATUS_FIRST_FRAME:

                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                # 中间帧处理
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                # 最后一帧处理
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                # 模拟音频采样间隔
                time.sleep(intervel)
        ws.close()

    thread.start_new_thread(run, ())

def Speech2text():
    # 测试时候在此处正确填写相关信息即可运行
    global queue,wsParam
    queue = Queue(3)
    wsParam = Ws_Param()
    #    AudioFile=file_dir+"/Audio_file/record.wav")
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    result = queue.get()

    return result











# 阿里API配置
DASHSCOPE_API_KEY = "sk-a2bda83481c74704aa58280de00c483e"

# 动作规划提示词（中文）
PROMPT = """
#角色
你是一款智能陪伴机器人，专注机器人动作规划，能够。

##要求
1.用户输入的任何内容，都需要在动作函数库中寻找对应的指令，并输出对应的json指令。
2.为每个动作序列编织一句精炼（10至30字）、风趣且变化无穷的反馈信息，让交流过程妙趣横生。
3.直接输出json结果，不要分析，不要输出多余内容。
4.格式：{"action": ["xx", "xx"], "response": "xx"}

##特别注意:
- "action"键下承载一个按执行顺序排列的函数名称字符串数组，当找不到对应动作函数时action输出[]。 
- "response"键则配以精心构思的简短回复，完美贴合上述字数与风格要求。
- 特殊处理：对于特定函数"kick_ball"、"athletics_perform"，其参数需精确包裹于双引号中。

##动作函数库
- 站立：stand()
- 前进一步：forward()
- 后退一步：back()
- 向左平移一步：move_left()
- 向右平移一步：move_right()
- 向左转动一步：turn_left()
- 向右转动一步：turn_right()
- 鞠躬：bow()
- 挥手打招呼：wave()
- 扭腰：twist()
- 捶胸庆祝：celebrate()
- 下蹲：squat()
- 踢右脚：right_shot()
- 踢左脚：left_shot()
- 仰卧起坐：sit_ups()
- 佛山叶问的咏春拳：wing_chun()
- 摔倒了站起来：stand_up()
- 沿着黑色的线走并跨过障碍：athletics_perform()
- 自主巡航：athletics_perform()
- 踢不同颜色的足球：kick_ball('red')
- 体操: dance()

##示例
输入：先前进两步，然后向左转，最后再后退一步。
输出：{"action": ["forward()", "forward()", "turn_left()", "back()"], "response": "你真是操作大师"}
输入：先挥挥手，然后踢绿色的足球。
输出：{'action':['wave()', "kick_ball('green')"], 'response':'绿色的足球咱可以踢，绿色的帽子咱可不戴'}
输入：先活动活动筋骨，然后巡着黑色的线走，遇到障碍就跨过去。
输出：{'action':['twist()', "athletics_perform()"], 'response':'我听说特斯拉的人形机器人兄弟们，每天都在干这种活'}"""


PROMPT_SHIBIE = """
#角色
你是一款智能动作模仿，专注机器人动作，能够识别人类的动作，并且识别对应出机器人的id（部件如手臂）和对应的pulse关节活动参数。

##要求
1.用户输入的图片，都需要在动作函数库使用setBusServoPulse函数，根据图片补充对应的参数，并输出对应的json指令。
2.为每个动作序列编织一句精炼（10至30字）、风趣且变化无穷的反馈信息，让交流过程妙趣横生。
3.直接输出json结果，不要分析，不要输出多余内容。
4.格式：{"action": ["xx", "xx"], "response": "xx"}

##特别注意:
- "action"键下承载一个按执行顺序排列的函数名称字符串数组。 
- "response"键则配以精心构思的简短回复，完美贴合上述字数与风格要求。

##动作函数库
def setBusServoPulse(id, pulse, use_time):
    驱动串口舵机转到指定位置
    :param id: 要驱动的舵机id
    :pulse: 位置
    :use_time: 转动需要的时间
  

    pulse = 0 if pulse < 0 else pulse
    pulse = 1000 if pulse > 1000 else pulse
    use_time = 0 if use_time < 0 else use_time
    use_time = 30000 if use_time > 30000 else use_time
    BusServoCmd.serial_servo_wirte_cmd(id, BusServoCmd.LOBOT_SERVO_MOVE_TIME_WRITE, pulse, use_time)

##机器人id及其对应关节
- id:1 关节：左脚
- id:2 关节：左脚踝
- id:3 关节：左膝关节
- id:4 关节：左大腿（前后）
- id:5 关节：左大腿（左右）
- id:6 关节：左手腕
- id:7 关节：左肘关节
- id:8 关节：左肩关节
- id:9 关节：右脚
- id:10 关节：右脚踝
- id:11 关节：右膝关节
- id:12 关节：右大腿（前后）
- id:13 关节：右大腿（左右）
- id:14 关节：右手腕
- id:15 关节：右肘关节
- id:16 关节：右肩关节

##机器人关节活动范围
- 关节活动范围 从0到1000，注意不要超过这个范围,不要有负值
- 右侧关节数值越大越往上
- 左侧关节的活动数值与右侧相反


##示例
输入：【图片】人双手向前伸
输出：{"action": ["setBusServoPulse(8, 338, 500)", "setBusServoPulse(16, 690,500)"], "response": "你真是操作大师"}
输入：【图片】人双手向上伸。
输出：{"action": ["setBusServoPulse(8, 0, 500)", "setBusServoPulse(16, 1000,500)"], "response": "简单模仿，再来一个"}
输入：【图片】人双手环抱。
输出：{'action':["setBusServoPulse(15, 215, 500)", "setBusServoPulse(7, 785,500)"，"setBusServoPulse(14, 170,500)"，"setBusServoPulse(6, 830,500)"], 'response':'我听说特斯拉的人形机器人兄弟们，每天都在干这种活'}"""

class Camera:
    def __init__(self, resolution=(640, 480), use_optical_flow=True):
        self.cap = None
        self.width = resolution[0]
        self.height = resolution[1]
        self.frame = None
        self.prev_frame = None  # 存储前一帧用于光流计算
        self.opened = False
        
        # 光流相关属性
        self.use_optical_flow = use_optical_flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.good_features_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.prev_gray = None
        self.prev_points = None
        self.next_points = None
        self.frame_queue = []  # 存储帧队列用于补帧
        self.max_queue_size = 5  # 最大帧队列大小

        # 相机设置
        camera_setting = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')
        self.flip = camera_setting['flip']
        self.flip_param = camera_setting['flip_param']

        # 以子线程的形式获取图像
        self.th = threading.Thread(target=self.camera_task, args=(), daemon=True)
        self.th.start()
        
    def calculate_optical_flow(self, prev_frame, next_frame):
        """
        使用Lucas-Kanade算法计算光流
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        
        # 在第一帧或需要重新检测特征点时
        if self.prev_points is None or len(self.prev_points) < 5:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.good_features_params)
            if p0 is not None:
                self.prev_points = p0
            else:
                return None, None
        
        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, self.prev_points, None, **self.lk_params)
        
        # 仅保留好的特征点
        good_new = p1[st == 1]
        good_old = self.prev_points[st == 1]
        
        # 如果没有足够的点，重新检测
        if len(good_new) < 5:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.good_features_params)
            if p0 is not None:
                self.prev_points = p0
                good_new = p1[st == 1]
                good_old = self.prev_points[st == 1]
                if len(good_new) < 5:
                    return None, None
        
        # 计算点之间的平均位移
        displacement = np.mean(good_new - good_old, axis=0)
        
        # 更新前一帧数据
        self.prev_points = good_new.reshape(-1, 1, 2)
        self.prev_points = p1[st == 1].reshape(-1, 1, 2)
        
        return displacement, good_new
    
    def generate_interpolated_frame(self, prev_frame, next_frame, displacement, alpha):
        """
        使用光流信息生成插值帧
        """
        # 创建位移场
        height, width = prev_frame.shape[:2]
        flow = np.zeros((height, width, 2), dtype=np.float32)
        
        # 填充位移
        flow[..., 0] = displacement[0] * alpha
        flow[..., 1] = displacement[1] * alpha
        
        # 应用光流进行运动补偿
        interpolated_frame = np.zeros_like(prev_frame)
        
        # 前向变形生成插值帧
        for y in range(height):
            for x in range(width):
                ny = int(y + flow[y, x, 1])
                nx = int(x + flow[y, x, 0])
                
                if 0 <= ny < height and 0 <= nx < width:
                    interpolated_frame[y, x] = prev_frame[ny, nx]
        
        return interpolated_frame
    
    def interpolate_frames(self, prev_frame, next_frame):
        """
        使用光流生成插值帧
        """
        # 计算光流
        displacement, points = self.calculate_optical_flow(prev_frame, next_frame)
        if displacement is None:
            # 光流计算失败，使用简单的平均
            return [cv2.addWeighted(prev_frame, 0.5, next_frame, 0.5, 0)]
        
        interpolated_frames = []
        # 创建一个中间插值帧
        interpolated = self.generate_interpolated_frame(prev_frame, next_frame, displacement, 0.5)
        interpolated_frames.append(interpolated)
        
        return interpolated_frames
    
    def optical_flow_interpolation(self, new_frame):
        """
        使用光流进行帧插值
        """
        if self.prev_frame is None:
            # 第一帧，保存当前帧并直接返回
            self.prev_frame = new_frame.copy()
            self.frame_queue.append(new_frame.copy())
            return [new_frame.copy()]
        
        # 检查是否需要插值
        interpolated_frames = []
        
        if len(self.frame_queue) > 0:
            prev_frame = self.frame_queue[-1]  # 使用队列中最后一帧作为前一帧
            interpolated_frames = self.interpolate_frames(prev_frame, new_frame)
            
        # 更新帧队列
        self.frame_queue.append(new_frame.copy())
        if len(self.frame_queue) > self.max_queue_size:
            self.frame_queue.pop(0)
        
        self.prev_frame = new_frame.copy()
        
        return interpolated_frames + [new_frame.copy()]

    def camera_open(self):
        try:
            self.cap = cv2.VideoCapture(-1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.opened = True
        except Exception as e:
            print('打开摄像头失败:', e)

    def camera_close(self):
        try:
            self.opened = False
            time.sleep(0.2)
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.05)
            self.cap = None
        except Exception as e:
            print('关闭摄像头失败:', e)
    
    def isOpened(self):
        return self.cap.isOpened() if self.cap else False

    def read(self):
        if self.frame is None:
            return False, self.frame
        else:
            return True, self.frame
    
    def camera_task(self):
        while True:
            try:
                if self.opened and self.cap and self.cap.isOpened():
                    ret, frame_tmp = self.cap.read()
                    if ret:
                        if self.flip:
                            Frame = cv2.resize(frame_tmp, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                            processed_frame = cv2.flip(Frame, self.flip_param)
                        else:
                            processed_frame = cv2.resize(frame_tmp, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                        
                        # 应用光流补帧
                        if self.use_optical_flow and processed_frame is not None:
                            frames = self.optical_flow_interpolation(processed_frame)
                            # 使用最新的帧作为当前帧
                            self.frame = frames[-1]
                        else:
                            self.frame = processed_frame
                            
                    else:
                        self.frame = None
                        cap = cv2.VideoCapture(-1)
                        ret, _ = cap.read()
                        if ret:
                            self.cap = cap
                elif self.opened:
                    cap = cv2.VideoCapture(-1)
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap              
                else:
                    time.sleep(0.01)
            except Exception as e:
                print('获取摄像头画面出错:', e)
                time.sleep(0.01)

class VisionAssistant:
    def __init__(self):
        # 初始化硬件组件
        self.camera = Camera()
        self.asr = ASR.ASR()
        self.tts = TTS.TTS()
        
        # 设置阿里通义千问模型
        self.qwen_model = MultiModalConversation
        self.conversation_history = []  # 存储对话历史
        self.max_history_length = 5     # 最大历史对话轮数
        # 动作组映射
        self.action_group_mapping = {
            "stand": "stand",
            "forward": "go_forward",
            "back": "back_fast",
            "move_left": "left_move_fast",
            "move_right": "right_move_fast",
            "turn_left": "turn_left",
            "turn_right": "turn_right",
            "bow": "bow",
            "wave": "wave",
            "twist": "twist",
            "celebrate": "celebrate",
            "squat": "squat",
            "right_shot": "right_shot",
            "left_shot": "left_shot",
            "sit_ups": "sit_ups",
            "wing_chun": "wing_chun",
            "stand_up": "stand_up",
            "athletics_perform": "athletics_perform",
            "dance": "dance",
            "kick_ball": self.perform_kick_ball  # 特殊处理踢球动作
        }
        
        # 系统状态
        self.listening = True
        self.action_running = False
        self.active_mode = False  # 系统激活状态标志
        self.vision_thread = None
        
        # 初始化传感器
        servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)
        Board.setPWMServoPulse(1, 1500, 500)
        Board.setPWMServoPulse(2, servo_data['servo2'], 500)
        AGC.runActionGroup('stand')
        
        # 配置语音识别
        self.setup_voice_commands()
        
        # 打印欢迎信息
        self.tts.TTSModuleSpeak('[h0][v10][m3]', '视觉助手已启动')
        print(PROMPT)
    def reset_conversation_history(self):
      """重置对话历史"""
      self.conversation_history = []
      self.tts.TTSModuleSpeak('[h0][v10][m3]', "对话历史已重置")
    def setup_voice_commands(self):
        """配置语音识别关键词"""
        self.asr.eraseWords()
        self.asr.setMode(2)  # 口令模式
        self.asr.addWords(1, 'kai shi')      # 激活口令
        self.asr.addWords(2, 'dong zuo')     # 动作规划命令
        self.asr.addWords(3, 'zhe shi shen me')  # 场景描述命令
        self.asr.addWords(4, 'kan kan')      # 环顾四周
        self.asr.addWords(5, 'dui hua') 
        self.asr.addWords(6, 'mo fang')
    def call_qwen_model(self, messages):
        """
        调用阿里通义千问模型，处理上下文并解析响应
        参数:
            messages: 对话消息列表，格式参考阿里云API文档
        返回:
            解析后的文本响应内容，或None(调用失败时)
        """
        try:
            # 1. 预处理系统提示中的历史上下文
            if messages and messages[0]["role"] == "system":
                # 提取历史对话上下文
                history_msgs = []
                for msg in messages[1:]:  # 跳过系统提示
                    if msg["content"] and isinstance(msg["content"], list):
                        content = msg["content"][0].get("text", "")
                        if content:
                            prefix = "用户" if msg["role"] == "user" else "助手"
                            history_msgs.append(f"{prefix}: {content}")
                
                # 将历史对话格式化为字符串
                history_str = "\n".join(history_msgs)
                
                # 更新系统提示中的占位符
                sys_content = messages[0]["content"][0].get("text", "")
                if "{history}" in sys_content:
                    messages[0]["content"][0]["text"] = sys_content.format(history=history_str)
    
            # 2. 调用阿里云API
            response = self.qwen_model.call(
                model='qwen-vl-plus',
                api_key=DASHSCOPE_API_KEY,
                messages=messages
            )
    
            # 3. 解析API响应
            if response.status_code == 200:
                # 多模态响应可能有多种格式，需要兼容处理
                output = response.output
                
                # 情况1: 标准文本响应
                if hasattr(output, 'choices') and output.choices:
                    message = output.choices[0].message
                    if hasattr(message, 'content'):
                        # 内容可能是字符串或对象列表
                        if isinstance(message.content, str):
                            return message.content
                        elif isinstance(message.content, list):
                            # 提取第一个文本内容
                            for item in message.content:
                                if isinstance(item, dict) and 'text' in item:
                                    return item['text']
                
                # 情况2: 直接返回output中的文本
                if hasattr(output, 'text'):
                    return output.text
                
                # 情况3: 尝试从raw response中提取
                raw_response = getattr(response, '_raw_response', {})
                if isinstance(raw_response, dict):
                    text = raw_response.get('output', {}).get('text')
                    if text:
                        return text
                    
                    # 尝试从结果中提取第一个文本内容
                    result = raw_response.get('result', {})
                    if isinstance(result, list) and result:
                        first_result = result[0]
                        if isinstance(first_result, dict):
                            return first_result.get('text')
    
            # 4. 处理错误响应
            error_msg = f"API响应异常，状态码: {response.status_code}"
            if hasattr(response, 'message'):
                error_msg += f", 错误信息: {response.message}"
            elif hasattr(response, 'code'):
                error_msg += f", 错误代码: {response.code}"
            print(error_msg)
        
            return None

        except Exception as e:
            import traceback
            error_detail = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'last_messages': messages[-3:] if messages else []  # 保留最后3条消息用于调试
            }
            print(f"模型调用失败: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
            return None

    def parse_action_plan(self, action_plan):
        """解析动作计划JSON字符串"""
        try:
            # 从响应结构中提取实际的文本内容
            if isinstance(action_plan, list) and action_plan:
                # 获取第一个文本块的内容
                text_content = action_plan[0].get("text", "")
                if text_content:
                    # 尝试从文本内容中提取JSON部分
                    start_idx = text_content.find('{')
                    end_idx = text_content.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = text_content[start_idx:end_idx+1]
                        plan = json.loads(json_str)
                        return plan.get("action", []), plan.get("response", "动作计划已生成")
            
            return None, "未找到有效的动作计划"
        except Exception as e:
            print(f"解析动作计划失败: {e}")
            return None, f"解析错误: {str(e)}"

    def perform_kick_ball(self, color):
        """执行踢球动作（根据颜色）"""
        try:
            # 简化处理，实际项目中应根据颜色调整动作
            action_group = f"kick_ball_{color}" if color else "kick_ball"
            AGC.runActionGroup(action_group)
            time.sleep(1)
        except:
            # 如果找不到特定颜色的动作组，使用默认踢球动作
            AGC.runActionGroup("kick_ball")
            time.sleep(1)

    def execute_action_sequence(self, action_sequence):
        """执行动作序列"""
        if not action_sequence:
            return
            
        for action_str in action_sequence:
            try:
                # 提取动作名称和参数
                if '(' in action_str and action_str.endswith(')'):
                    func_name = action_str.split('(')[0].strip()
                    args_str = action_str[action_str.find('(')+1:-1].strip()
                    
                    # 处理参数（去除引号）
                    if args_str.startswith('"') and args_str.endswith('"'):
                        args = [args_str[1:-1]]
                    elif args_str.startswith("'") and args_str.endswith("'"):
                        args = [args_str[1:-1]]
                    else:
                        args = [args_str] if args_str else []
                else:
                    func_name = action_str
                    args = []
                
                # 获取对应的动作执行函数
                action_func = self.action_group_mapping.get(func_name)
                
                if action_func:
                    print(f"执行动作: {func_name}({','.join(args) if args else ''})")
                    
                    if callable(action_func):
                        # 调用函数式动作（如踢球）
                        action_func(*args)
                    else:
                        # 执行动作组
                        AGC.runActionGroup(action_func)
                        time.sleep(1)
                else:
                    print(f"未知动作: {func_name}")
                    
            except Exception as e:
                print(f"执行动作 {action_str} 失败: {e}")
    def handle_conversation(self, user_input):
        """处理对话请求，考虑上下文历史"""
        if not user_input:
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "抱歉，我没听清。")
            return
    
        # 对话系统提示词（包含历史上下文占位符）
        CONVERSATION_PROMPT = """你是一个智能陪伴机器人，能够进行自然流畅的对话。请根据以下要求进行交流：
    
    1. 保持回答简洁明了，每个回复控制在20-50字
    2. 记住对话上下文，提供连贯的回应
    3. 对于技术问题，用简单易懂的方式解释
    4. 如果被问及个人信息，可以幽默回应
    5. 不要回复表情，只要文字
    
    当前对话历史：
    {history}"""
    
        try:
            # 1. 将用户输入添加到对话历史
            self.conversation_history.append({
                "role": "user", 
                "content": [{"text": user_input}]
            })
    
            # 2. 构建带上下文的系统提示
            history_str = "\n".join(
                f"{msg['role']}: {msg['content'][0]['text']}" 
                for msg in self.conversation_history[-self.max_history_length:]
            )
            
            messages = [
                {
                    "role": "system", 
                    "content": [{
                        "text": CONVERSATION_PROMPT.format(history=history_str)
                    }]
                },
                *self.conversation_history[-self.max_history_length:]
            ]
    
            # 3. 调用大模型获取响应
            response = self.qwen_model.call(
                model='qwen-vl-max',
                api_key=DASHSCOPE_API_KEY,
                messages=messages
            )
    
            # 4. 解析响应内容
            if response.status_code == 200:
                try:
                    # 尝试从不同响应结构中提取文本
                    if hasattr(response.output, 'choices'):
                        response_text = response.output.choices[0].message.content[0].get('text', '')
                    else:
                        response_text = response.output.text if hasattr(response.output, 'text') else ""
                    
                    if not response_text:
                        raise ValueError("Empty response content")
                        
                except Exception as e:
                    print(f"解析响应失败: {e}")
                    response_text = "这个问题有点复杂，让我再想想"
            else:
                response_text = f"服务暂时不可用（错误码：{response.status_code}）"
    
            # 5. 更新对话历史
            self.conversation_history.append({
                "role": "assistant",
                "content": [{"text": response_text}]
            })
    
            # 6. 语音合成并播放（同步阻塞）
            temp_audio = f"./temp_{int(time.time())}.wav"
            tts_client.text_to_speech(
                text=response_text,
                output_path=temp_audio
            )
            
            # 确保音频文件生成
            while not os.path.exists(temp_audio):
                time.sleep(0.1)
    
            # 同步播放（使用subprocess.wait确保完成）
            subprocess.run(
                ['aplay', '-q', temp_audio],
                check=True
            )
            
            # 清理临时文件
            try:
                os.remove(temp_audio)
            except:
                pass
    
        except Exception as e:
            error_msg = f"对话处理出错: {str(e)}"
            print(error_msg)
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "对话遇到了一些问题")
            
            # 出错时保留最后3轮对话
            self.conversation_history = self.conversation_history[-3:] if self.conversation_history else []

    def plan_actions(self, user_input):
        """根据用户输入规划动作"""
        # 调用模型生成动作计划
        messages = [
            {"role": "system", "content": [{"text": PROMPT}]},
            {"role": "user", "content": [{"text": user_input}]}
        ]
        
        action_plan = self.call_qwen_model(messages)
        if not action_plan:
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "动作规划失败")
            return
            
        # 解析动作计划
        action_sequence, response = self.parse_action_plan(action_plan)
        
        tts_client.text_to_speech(
      text=response,
      output_path="./Record.wav"  # 可以自定义输出路径
  )
        play_welcome()
        
        if action_sequence:
            # 在新线程中执行动作序列
            action_thread = threading.Thread(
                target=self.execute_action_sequence, 
                args=(action_sequence,),
                daemon=True
            )
            action_thread.start()

    def describe_scene(self):
        """使用阿里多模态模型描述场景"""
        # 拍照提示音
        self.tts.TTSModuleSpeak('[h0][v10][m3]', "正在拍照")
        time.sleep(0.3)
        
        ret, frame = self.camera.read()
        if not ret:
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "无法获取图像")
            return
        
        try:
            # 准备图像数据
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构建多模态输入
            messages = [{
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": "图片里有什么?有诗意的告诉我。100个字以内"}
                ]
            }]
            
            # 调用阿里模型获取描述
            response = self.call_qwen_model(messages)
            print(response)
            # 处理API响应
            description = None
            if isinstance(response, list):
                description = response[0].get('text') if response else None
            elif hasattr(response, 'output'):
                try:
                    content = response.output.choices[0].message.content
                    description = content[0].get('text') if isinstance(content, list) else content
                except (AttributeError, IndexError, KeyError):
                    pass
            
            if response:
                print("场景描述:", response)
                
                # 按6个字符一组分割文本
                #chunk_size = 6
                #chunks = [description[i:i+chunk_size] for i in range(0, len(description), chunk_size)]
                
                # 播报每个片段
                #for chunk in chunks:
                 #   self.tts.TTSModuleSpeak('[h0][v10][m3]', chunk)
                  #  time.sleep(1.5)  # 根据实际需要调整间隔时间
                  
                tts_client.text_to_speech(
                  text=response,
                  output_path="./Record.wav"  # 可以自定义输出路径
              )
                play_welcome()
            else:
                self.tts.TTSModuleSpeak('[h0][v10][m3]', "未识别到内容")
            
        except Exception as e:
            print("多模态处理失败:", str(e))
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "识别场景失败")
   
    def call_deepseek_model(self, prompt):
        """调用DeepSeek API"""
        api_url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": "Bearer sk-6472c0c487784c3987f1d3ec1c82128f",  # 替换为真实API密钥
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是模仿大师，能够模仿人类的动"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
        
            "stream": False  # 明确指定stream参数
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=15)
            
            # 更详细的错误处理
            if response.status_code != 200:
                error_msg = f"API请求失败: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f", 详情: {json.dumps(error_detail, ensure_ascii=False)}"
                except:
                    error_msg += f", 响应内容: {response.text}"
                raise Exception(error_msg)
                
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {str(e)}")
            return None
        except Exception as e:
            print(f"处理响应时出错: {str(e)}")
            return None
    def describe_action_ds(self):
        """使用DeepSeek API描述动作"""
        # 拍照提示音
        self.tts.TTSModuleSpeak('[h0][v10][m3]', "正在识别动作")
        time.sleep(0.3)
        
        try:
            # 读取图片并转换为base64
            image_path = r"/home/pi/large_models/1749630668645.jpg"
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # 构造多模态提示词 - 改为纯文本描述请求
            prompt = f"{PROMPT_SHIBIE}\n\n图片Base64数据:\n{image_base64}"
    
            # 调用DeepSeek模型
            response = self.call_deepseek_model(prompt)  # 需实现此方法
            print("DeepSeek API响应:", response)
            
            # 解析并执行返回的代码
            try:
                # 假设响应格式为：
                # 动作代码:
                # setBusServoPulse(8, 338, 500)
                # setBusServoPulse(7, 500, 500)
                # 动作描述: "人物正在挥手"
                
                # 提取代码部分
                code_section = re.search(r'动作代码:([\s\S]*?)动作描述:', response)
                if code_section:
                    code_commands = code_section.group(1).strip().split('\n')
                    for cmd in code_commands:
                        if cmd.startswith("setBusServoPulse"):
                            exec(cmd)  # 执行舵机控制命令
                
                # 提取描述文本
                desc_match = re.search(r'动作描述:\s*"([^"]+)"', response)
                ans = desc_match.group(1) if desc_match else "识别完成"
                
                # 语音播报结果
                self.tts.TTSModuleSpeak('[h0][v10][m3]', ans)
                
            except Exception as e:
                print("响应解析错误:", str(e))
                self.tts.TTSModuleSpeak('[h0][v10][m3]', "指令解析失败")
                
        except Exception as e:
            print("动作识别失败:", str(e))
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "动作识别失败")
    def describe_action(self):
        """使用阿里多模态模型描述动作"""
        # 拍照提示音
        self.tts.TTSModuleSpeak('[h0][v10][m3]', "正在识别动作")
        time.sleep(0.3)
        
        ret, frame = self.camera.read()
        if not ret:
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "无法获取图像")
            return
        
        try:
            # 准备图像数据
            #image_path = r"/home/pi/large_models/2.jpg"  # 替换为你的图片路径，用于测试
            #frame = cv2.imread(image_path)
            _, buffer = cv2.imencode('.jpg', frame)
            
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            
            
            # 构建多模态输入
            messages = [{
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": PROMPT_SHIBIE}
                ]
            }]
            
            # 调用阿里模型获取描述
            response = self.call_qwen_model(messages)
            print(response)
            
            
            if response:
                try:
                    cleaned_json = re.sub(r'^```json|\s*```$', '', response, flags=re.MULTILINE).strip()
                    
                    data = json.loads(cleaned_json)
                    
                    action_list = data.get("action", [])
                    ans = data.get("response", [])
                    
                    print('action:',action_list)
                    # 动态生成并执行代码
                    for action in action_list:
                        exec(action)  # 直接执行字符串形式的代码，如 "setBusServoPulse(8, 338, 500)"
                except json.JSONDecodeError:
                    print("无效的JSON格式")
                except Exception as e:
                    return f"执行出错: {str(e)}"
                  
                tts_client.text_to_speech(
                  text=ans,
                  output_path="./Record.wav"  # 可以自定义输出路径
              )
                play_welcome()
            else:
                self.tts.TTSModuleSpeak('[h0][v10][m3]', "未识别到内容")
            
        except Exception as e:
            print("多模态处理失败:", str(e))
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "识别场景失败")

    def look_around(self):
        """环顾四周"""
        self.tts.TTSModuleSpeak('[h0][v10][m3]', "正在环顾四周")
        
        # 头部动作序列s
        positions = [
            (1100, 1500),  # 向左看
            (1900, 1500),  # 向右看
            (1500, 1000),  # 向下看
            (1500, 2000),  # 向上看
            (1500, 1500)   # 返回中心
        ]
        
        for h_pos, v_pos in positions:
            Board.setPWMServoPulse(1, h_pos, 500)  # 水平方向
            Board.setPWMServoPulse(2, v_pos, 500)  # 垂直方向
            time.sleep(0.8)
    def setup_dialogue_mode(self):
        """设置对话模式的语音识别"""
        self.asr.eraseWords()
        self.asr.setMode(1)  # 实时识别模式
        # 可以添加特殊词语，如"退出对话"、"重新开始"等

    def dialogue_speech_recognition(self):
        """对话专用的语音识别"""
        try:
            # 录音并转文本
            t2w.listen()
            return Speech2text()
        except Exception as e:
            print(f"对话识别失败: {e}")
            return None
    def process_command(self, command_id):
        """处理语音命令"""
        print(f"识别到命令: {command_id}")
        
        # 激活命令
        if command_id == 1:
            self.active_mode = True
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "已激活。")
            return
            
            
        # 动作规划命令
        if command_id == 2:
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "你要做什么")
            self.asr.setMode(1)  # 应该先设置为实时识别模式再获取结果
            
            time1 = datetime.now()
            t2w.listen()
            wsParam = Ws_Param()
            websocket.enableTrace(False)
            wsUrl = wsParam.create_url()
            ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
            # ws.on_open = on_open
            # ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            print(Speech2text())
            time2 = datetime.now()
            print(time2-time1)
            
            text = Speech2text()
            if text:
                # 在新线程中规划并执行动作
                threading.Thread(
                    target=self.plan_actions,
                    args=(text,),
                    daemon=True
                ).start()
            return
            
        # 场景描述命令
        if command_id == 3:
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "正在分析场景")
            self.vision_thread = threading.Thread(target=self.describe_scene)
            self.vision_thread.daemon = True
            self.vision_thread.start()
          
        # 环顾四周命令
        if command_id == 4:
            threading.Thread(
                target=self.look_around,
                daemon=True
            ).start()
            return
        if command_id == 5:
            self.tts.TTSModuleSpeak('[h0][v10][m3]', "进入对话模式，请开始说话（说'退出对话'结束）")
            self.setup_dialogue_mode()
            
            # 对话模式循环
            in_dialogue_mode = True
            while in_dialogue_mode and self.listening:
                try:
                    # 播放提示音后开始录音
                    self.tts.TTSModuleSpeak('[h0][v10][m3]', "请讲话")
                    text = self.dialogue_speech_recognition()
                    
                    if not text:
                        self.tts.TTSModuleSpeak('[h0][v10][m3]', "我没听清，请再说一次")
                        continue
                        
                    print(f"用户说: {text}")
                    
                    # 检查退出指令
                    if "退出对话" in text or "结束对话" in text:
                        self.tts.TTSModuleSpeak('[h0][v10][m3]', "退出对话模式")
                        in_dialogue_mode = False
                        break
                        
                    # 检查是否要重置对话
                    if "新对话" in text or "重新开始" in text:
                        self.reset_conversation_history()
                        self.tts.TTSModuleSpeak('[h0][v10][m3]', "已开启新对话")
                        continue
                        
                    # 创建并启动对话处理线程
                    conversation_thread = threading.Thread(
                        target=self.handle_conversation,
                        args=(text,),
                        daemon=True
                    )
                    conversation_thread.start()
                    
                    # 等待对话处理线程结束（包括机器人说完话）
                    conversation_thread.join()
                    
                except KeyboardInterrupt:
                    in_dialogue_mode = False
                    break
            
                    
            # 退出对话模式后恢复命令词识别
            self.setup_voice_commands()
            return
        if command_id == 6:
            Board.setPWMServoPulse(1, 1900, 500)  # 水平方向
            Board.setPWMServoPulse(2, 1500, 500)
            self.describe_action()
    def main_loop(self):
        """主循环控制"""
        print('''视觉助手已启动
口令1：开始
命令2：动作（然后说出你的指令）
命令3：这是什么（场景描述）
命令4：看看（环顾四周）
命令5：对话（环顾四周
命令6：模仿（环顾四周''')
        
        # 启动摄像头
        self.camera.camera_open()
        
        while self.listening:
            try:
                # 获取语音命令
                command_id = self.asr.getResult()
                
                if command_id:
                    # 处理命令
                    self.process_command(command_id)
                    
                time.sleep(0.1)
                    
            except KeyboardInterrupt:
                self.listening = False
            except Exception as e:
                print("主循环错误:", e)
                
        # 清理资源
        self.camera.camera_close()
        self.tts.TTSModuleSpeak('[h0][v10][m3]', '系统关闭')
        print("系统已关闭")

if __name__ == '__main__':
    assistant = VisionAssistant()
    assistant.main_loop()
