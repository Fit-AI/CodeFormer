#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
RUN rm /etc/apt/sources.list.d/cuda.list

RUN python3 -c "import torch; assert torch.cuda.is_available()"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&                   \
    apt-get install -y build-essential  \
                       python3          \
                       python3-pip      \
                       libopencv-dev

WORKDIR /app
ADD requirements.txt /app

RUN pip install --upgrade pip && 	      \
    pip install numpy --upgrade &&  	  \
    pip install -r requirements.txt

COPY weights /app/weights/
COPY basicsr /app/basicsr/
COPY README.md /app
RUN python basicsr/setup.py develop

COPY facelib /app/facelib

COPY service.py /app
COPY run.sh /app

ARG SUPABASE_URL_ARG
ENV SUPABASE_URL=$SUPABASE_URL_ARG

ARG SUPABASE_KEY_ARG
ENV SUPABASE_KEY=$SUPABASE_KEY_ARG

RUN apt-get install -y ffmpeg
RUN pip install ffmpeg-python

CMD ["./run.sh"]
