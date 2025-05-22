FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ARG MODEL_NAME
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools


# Install the requirements
# COPY --chown=user:user requirements.txt .
# RUN python -m pip install --user -r requirements.txt
RUN git clone https://github.com/ntnu-mr-cancer/dragon_baseline.git
RUN cd dragon_baseline && python -m pip install --user -r requirements.txt && python -m pip install --user -e . && cd ..

# Download the model, tokenizer and metrics
COPY --chown=user:user download_model.py .
# Download the model you want to use, e.g.:
RUN python download_model.py --model_name ${MODEL_NAME}
COPY --chown=user:user download_metrics.py .
RUN python download_metrics.py

# Set the environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV HF_EVALUATE_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

RUN mkdir -p /opt/app/FedDragon
COPY . /opt/app/FedDragon

WORKDIR /opt/app/FedDragon
RUN python -m pip install -r requirements.txt \
    && rm -rf /home/user/.cache/pip

ENTRYPOINT [ "python", "-m", "process" ]