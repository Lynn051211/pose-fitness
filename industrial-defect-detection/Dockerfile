# 工业表面缺陷检测系统 Docker 镜像
# 构建: docker build -t defect-detector .
# 运行: docker run --rm defect-detector python main.py --inference-only data/raw/val/images/defect_0000.jpg

FROM python:3.10-slim

# 系统依赖（OpenCV 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制全部项目文件
COPY . .

# 创建输出目录
RUN mkdir -p output models data

# 默认命令：运行推理（需挂载模型和图像）
CMD ["python", "main.py", "--inference-only", "data/raw/val/images/defect_0000.jpg"]
