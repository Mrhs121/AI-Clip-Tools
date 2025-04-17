# AI-Clip-Tools
AI辅助视频剪辑工具

# 一、通过AI自动剪辑出只包含人像的片段
使用场景：例如监控摄像头中包含大量的无用片段，通过yolo来识别人脸，只保留有人脸的片段，或者从影视作品中抽取包含人物的素材片段等

# Requirements

+ `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`


+ `pip install ultralytics opencv-python ffmpeg-python tqdm`

+ 原生ffmpeg可执行环境

# 用法

直接 python detect.py 即可

# Demo
+  demo.mp4 ： 样例输入视频 （github100m限制，自行访问网盘下载 https://pan.quark.cn/s/ca75a17519f2）

+  demo_output_video.mp4： 样例输出完整视频

+  output_clips ： 样例输出包含人物的视频小片段
