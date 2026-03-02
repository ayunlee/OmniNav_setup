# =============================================================================
# OmniNav GPU Dockerfile - aarch64 (ARM64) + CUDA 13.0
# =============================================================================
FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# 병렬 빌드 설정
ARG BUILD_JOBS=0
RUN CORES=$(nproc) && \
    if [ "$BUILD_JOBS" = "0" ] || [ -z "$BUILD_JOBS" ]; then \
        BUILD_JOBS=$(python3 -c "import math; print(max(2, int($CORES * 0.9)))"); \
    fi && \
    echo "export BUILD_JOBS=$BUILD_JOBS" >> /root/.bashrc && \
    echo "$BUILD_JOBS" > /tmp/build_jobs.txt

# 미러 설정 (Ubuntu Ports -> Kakao)
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirror.kakao.com/ubuntu-ports|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl vim ninja-build pkg-config \
    libgl1-mesa-dev libglu1-mesa-dev libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libegl1 libegl-dev libxrandr2 libxss1 libxcursor1 libxinerama1 libxi6 \
    libpangocairo-1.0-0 libatk1.0-0 libcairo-gobject2 libgtk-3-0 libgdk-pixbuf2.0-0 libglfw3-dev \
    locales software-properties-common gnupg lsb-release \
    libssl-dev libusb-1.0-0-dev libgtk-3-dev \
    iproute2 can-utils \
    gedit \
    && (apt-get install -y libasound2t64 || apt-get install -y libasound2 || true) \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    numpy==1.26.4 opencv-python-headless pillow matplotlib scipy tqdm omegaconf hydra-core numba \
    qwen-vl-utils==0.0.10 accelerate datasets safetensors sentencepiece einops peft trl==0.24.0 diffusers \
    "modelscope[datasets]>=1.19" "gradio>=4.0.0" fastapi uvicorn tensorboard pandas nltk rouge requests \
    aiohttp addict attrdict dacite charset_normalizer binpacking importlib_metadata jieba openai oss2 \
    tiktoken jsonlines transformers_stream_generator zstandard blake3 "pydantic>=2.0.0" "pydantic_core>=2.0.0"

RUN pip install --no-cache-dir deepspeed || echo "skip deepspeed"
RUN pip install --no-cache-dir liger-kernel || echo "skip liger"
RUN pip install --no-cache-dir --upgrade triton || echo "skip triton"
RUN pip install --no-cache-dir open3d || echo "skip open3d"
RUN pip install --no-cache-dir cpm_kernels || echo "skip cpm_kernels"

WORKDIR /workspace
COPY train_code/transformers-main /workspace/train_code/transformers-main
RUN cd /workspace/train_code/transformers-main && pip install --no-cache-dir -e . || true

# =============================================================================
# ROS2 Jazzy Jalisco 설치 (수정됨: Curl 방식 + GPG Temp 에러 해결)
# =============================================================================
ENV ROS_DISTRO=jazzy
ENV ROS_PYTHON_VERSION=3

# 1. GPG 에러 방지를 위한 디렉토리 생성 및 Curl 기반 키 다운로드
RUN mkdir -p /root/.gnupg && chmod 700 /root/.gnupg && \
    apt-get update && \
    apt-get install -y software-properties-common curl gnupg2 lsb-release && \
    add-apt-repository universe && \
    # 키 다운로드 (GitHub -> 실패시 원본 -> GPG 변환)
    (curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg || \
     curl -sSL https://gitee.com/ohhu/rosdistro/raw/master/ros.asc | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg) && \
    # 저장소 추가 (칭화대 미러)
    # (공식 ROS 저장소 - 가장 안정적)
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 2. ROS2 패키지 설치
RUN apt-get update && \
    apt-get install -y \
    ros-jazzy-ros-base \
    ros-jazzy-compressed-image-transport \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# 3. 빌드 도구 설치 (Pip 사용)
RUN pip install --no-cache-dir colcon-common-extensions vcstool

# 4. rosdep 초기화 (실패해도 빌드는 계속되도록 처리)
RUN rosdep init || true && \
    rosdep update || echo "rosdep update failed, please run manually"

# 5. 환경 변수
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "export ROS_DISTRO=jazzy" >> /root/.bashrc

RUN mkdir -p /workspace/src
# =============================================================================
# ROS2 Control 및 추가 의존성
# =============================================================================
RUN apt-get update && apt-get install -y \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager \
    ros-jazzy-twist-mux \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-diff-drive-controller \
    && rm -rf /var/lib/apt/lists/*


# =============================================================================
# Scout Mini ROS2 설치 및 빌드 (수정됨)
# =============================================================================
WORKDIR /workspace/src

# 1. 소스 코드 다운로드 및 의존성 import
# (vcstool, rosdep은 위에서 이미 설치했으므로 중복 설치 제거)
RUN git clone https://github.com/roasinc/scout_mini_ros2.git

# 패키지 목록 업데이트, vcstool 설치, 캐시 삭제를 한 번의 RUN 명령으로 수행
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-vcstool \
    ros-jazzy-compressed-image-transport \
    ros-jazzy-realsense2-camera \
    ros-jazzy-image-transport-plugins


RUN vcs import . < scout_mini_ros2/requirement.rosinstall

WORKDIR /workspace

# 2. rosdep 의존성 설치 (핵심 수정: apt-get update 추가)
# 이전 레이어에서 apt list를 지웠기 때문에, rosdep 실행 전 반드시 update가 필요합니다.
RUN apt-get update && \
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y

# 1. 기본 쉘을 bash로 변경 (source 명령어를 인식하기 위함)
SHELL ["/bin/bash", "-c"]

# 2. 빌드 명령어 실행 (한 줄로 연결)
# 중간에 source install/setup.bash가 포함되어 있어, 
# 두 번째 빌드 명령어가 첫 번째 빌드 결과를 인식할 수 있음
RUN source /opt/ros/jazzy/setup.bash && \
    rm -rf docker-examples/ tutorials/ && \
    colcon build --packages-select scout_mini_msgs && \
    source install/setup.bash && \
    colcon build --symlink-install --packages-ignore scout_mini_msgs

# 2. bash 실행 시마다 자동으로 source 하도록 .bashrc에 추가
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc

WORKDIR /workspace/OmniNav
CMD ["/bin/bash"]
