以下是关于 **edge-tts** 的详细介绍，包括安装、基本使用、高级功能以及应用场景，帮助您快速掌握这个强大的文本转语音工具。

---

## **1. 什么是 edge-tts？**
**edge-tts** 是一个基于 Python 的文本转语音（TTS）库，它利用微软 Edge 浏览器的在线语音合成服务，支持多种语言和声音风格。它的主要特点包括：
- **免费开源**：无需 API 密钥，可直接使用 。
- **多语言支持**：包括中文（普通话、粤语、台湾腔等）、英语、法语、德语等 。
- **高自然度**：采用微软 Azure 的神经网络 TTS 技术，语音流畅自然 。
- **命令行 & Python API**：既可以直接在终端使用，也能集成到 Python 代码中 。

---

## **2. 安装 edge-tts**
### **方法 1：pip 安装**
```bash
pip install edge-tts
```
安装完成后，可通过以下命令测试是否成功：
```bash
edge-tts --version
```

### **方法 2：源码安装（可选）**
如需最新版本，可从 GitHub 克隆：
```bash
git clone https://github.com/rany2/edge-tts.git
cd edge-tts
pip install .
```

---

## **3. 基本使用**
### **（1）命令行模式**
#### **① 生成语音文件**
```bash
edge-tts --text "你好，世界！" --write-media hello.mp3
```
- `--text`：要转换的文本。
- `--write-media`：输出音频文件路径 。

#### **② 选择语音风格**
查看所有可用语音：
```bash
edge-tts --list-voices
```
示例（使用中文男声）：
```bash
edge-tts --voice zh-CN-YunxiNeural --text "大家好，欢迎使用 edge-tts" --write-media output.mp3
```

#### **③ 调整语速和音量**
```bash
edge-tts --rate=+20% --volume=-10% --text "语速加快，音量降低" --write-media adjusted.mp3
```
- `--rate`：语速（`+` 加快，`-` 减慢）。
- `--volume`：音量调整 。

#### **④ 实时播放（不保存文件）**
```bash
edge-playback --text "直接播放这段语音"
```
（需安装 `mpv` 播放器）。

---

### **（2）Python API 模式**
#### **① 基本文本转语音**
```python
import edge_tts
import asyncio

async def text_to_speech():
    voice = "zh-CN-YunxiNeural"  # 中文男声
    text = "欢迎使用 edge-tts 进行语音合成"
    output_file = "output.mp3"
    
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_file)

asyncio.run(text_to_speech())
```

#### **② 动态选择语音**
```python
from edge_tts import VoicesManager

async def dynamic_voice():
    voices = await VoicesManager.create()
    chinese_voices = voices.find(Language="zh-CN", Gender="Female")  # 查找中文女声
    selected_voice = chinese_voices[0]["Name"]
    
    communicate = edge_tts.Communicate(text="你好！", voice=selected_voice)
    await communicate.save("hello.mp3")

asyncio.run(dynamic_voice())
```

#### **③ 流式处理（适用于长文本）**
```python
async def stream_audio():
    communicate = edge_tts.Communicate(text="这是一段较长的文本...", voice="zh-CN-YunyangNeural")
    with open("long_audio.mp3", "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

asyncio.run(stream_audio())
```

---

## **4. 高级功能**
### **（1）生成字幕（VTT 格式）**
```bash
edge-tts --text "Hello, world!" --write-media hello.mp3 --write-subtitles hello.vtt
```
适用于视频配音时同步字幕 。

### **（2）批量处理文本文件**
```python
import edge_tts
import asyncio

async def batch_convert():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    communicate = edge_tts.Communicate(text=text, voice="zh-CN-YunxiNeural")
    await communicate.save("batch_output.mp3")

asyncio.run(batch_convert())
```

### **（3）调整音高（Pitch）**
```bash
edge-tts --pitch=+50Hz --text "音调提高" --write-media high_pitch.mp3
```
（仅部分语音支持）。

---

## **5. 应用场景**
1. **视频配音**：自动生成解说音频 。
2. **电子书朗读**：将文章转为语音方便收听。
3. **智能助手**：为聊天机器人增加语音交互 。
4. **语言学习**：模仿不同地区的口音（如台湾腔、陕西话）。

---

## **6. 常见问题**
### **Q1：安装失败？**
- 确保 Python ≥ 3.6，并更新 pip：
  ```bash
  python -m pip install --upgrade pip
  ```

### **Q2：无法播放音频？**
- 安装 `mpv` 播放器：
  ```bash
  # macOS/Linux
  brew install mpv
  # Windows（使用 Chocolatey）
  choco install mpv
  ```

### **Q3：如何商用？**
- edge-tts 是免费开源的，但需遵守微软 Azure TTS 的使用政策 。

---

## **7. 总结**
**edge-tts** 是一款功能强大且易用的 TTS 工具，特别适合需要 **免费、高自然度语音合成** 的用户。无论是通过命令行快速生成语音，还是用 Python 进行高级集成，它都能满足需求。

**推荐用途**：
- 个人项目、自动化脚本、教育用途。
- 替代部分收费 TTS 服务（如百度语音、讯飞）。

**GitHub 地址**：[https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts) 。

如果有更具体的需求（如方言支持、长文本优化），欢迎进一步探讨！ 🚀
