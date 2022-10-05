FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV AWS_CREDENTIALS=W2RlZmF1bHRdCmF3c19hY2Nlc3Nfa2V5X2lkID0gQUtJQVJTSDJFVlZZRFlUQTQzS04KYXdzX3NlY3JldF9hY2Nlc3Nfa2V5ID0gV0M0MjhhUm12eDhMYTVDV2ExSU9oM3RwRnFqU2g0QU1iN0FOQ2tKWQo=
ENV AWS_CONFIG=W2RlZmF1bHRdCnJlZ2lvbiA9IHVzLWVhc3QtMg==

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl ffmpeg libsm6 libxext6


ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment


RUN conda install -c conda-forge cudatoolkit=11.6 && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ && \

    # Installing python dependencies
    RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

RUN pip3 --timeout=300 --no-cache-dir install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

COPY ./requirements.txt .
RUN pip3 --timeout=300 --no-cache-dir install -r requirements.txt

# Copy model files
COPY ./models /models

# Copy app files
COPY ./app /app
WORKDIR /app/
ENV PYTHONPATH=/app
RUN ls -lah /app/* && mkdir ~/.aws

RUN echo "${AWS_CREDENTIALS}" | base64 --decode > ~/.aws/credentials && echo "${AWS_CONFIG}" | base64 --decode > ~/.aws/config

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 80
# CMD ["/start.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
