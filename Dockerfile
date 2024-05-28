# app/Dockerfile

FROM python:3.10.6
EXPOSE 8080

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/Nsayre/helmet_det_streamlit .
WORKDIR /app
COPY . ./

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 49152-65535

ENTRYPOINT ["streamlit", "run", "app/run_faster_cache.py", "--server.port=8080", "--server.address=0.0.0.0"]
