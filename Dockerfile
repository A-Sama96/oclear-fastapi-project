FROM python:3.9.14-slim

ENV AWS_CREDENTIALS=W2RlZmF1bHRdCmF3c19hY2Nlc3Nfa2V5X2lkID0gQUtJQVJTSDJFVlZZRFlUQTQzS04KYXdzX3NlY3JldF9hY2Nlc3Nfa2V5ID0gV0M0MjhhUm12eDhMYTVDV2ExSU9oM3RwRnFqU2g0QU1iN0FOQ2tKWQo=
ENV AWS_CONFIG=W2RlZmF1bHRdCnJlZ2lvbiA9IHVzLWVhc3QtMg==

# install utilities
RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get install --no-install-recommends -y curl \
    && apt-get clean

COPY ./requirements.txt .
RUN pip --timeout=300 --no-cache-dir install -r requirements.txt
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
# Copy model files
COPY ./models /models

# Copy app files
COPY ./app /app
WORKDIR /app/
ENV PYTHONPATH=/app
RUN ls -lah /app/* && mkdir ~/.aws

RUN echo "${AWS_CREDENTIALS}" | base64 --decode > ~/.aws/credentials && echo "${AWS_CONFIG}" | base64 --decode > ~/.aws/config

# COPY ./start.sh /start.sh
# RUN chmod +x /start.sh
# CMD ["/start.sh"]
# EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
