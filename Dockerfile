# 1. 베이스 이미지로 Python 3.9 사용
FROM python:3.9-slim

# 2. 작업 디렉토리 생성 및 이동
WORKDIR /app

# 3. 필요한 파일 다운로드
RUN apt-get update && apt-get install -y curl
RUN curl -L "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -o "GroundingDINO/weights/groundingdino_swint_ogc.pth" && \
    curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -o "GroundingDINO/weights/sam_vit_h_4b8939.pth"

# 4. 소스 파일 복사 (로컬의 모든 파일을 컨테이너로 복사)
COPY . /app

# 5. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 6. FastAPI 서버 실행
CMD ["uvicorn", "api-server:app", "--host", "0.0.0.0", "--port", "8000"]
