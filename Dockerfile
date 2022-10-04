FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

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

#COPY models/ /app/models
#COPY options/ /app/options
#COPY util/ /app/util
#COPY insightface_func/ /app/insightface_func
#COPY parsing_model/ /app/parsing_model
#COPY simswaplogo/ /app/simswaplogo
#COPY arcface_model/ /app/arcface_model
#COPY checkpoints/ /app/checkpoints
#RUN pip3 install --upgrade pip && pip install gunicorn munch supabase
#COPY service.py /app
#COPY run.sh /app
#COPY default_configuration.json /app/

#ARG SUPABASE_URL_ARG
#ENV SUPABASE_URL=$SUPABASE_URL_ARG

#ARG SUPABASE_KEY_ARG
#ENV SUPABASE_KEY=$SUPABASE_KEY_ARG

#CMD ["./run.sh"]
