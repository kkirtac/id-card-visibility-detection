FROM nvcr.io/nvidia/pytorch:21.09-py3

# Set working directory
WORKDIR /app

COPY . .

# Install additional system packages and clean up apt cache aftwards
RUN apt-get update && \
    apt-get install -y screen sudo && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r /app/requirements.txt

RUN echo "Starting Jupyter notebook..." && \
    jupyter notebook --port=8888 --ip=0.0.0.0 --no-browser . &

RUN echo "Starting tensorboard..." && \
    tensorboard --logdir=/app/tb_logs/my_visibility_model
