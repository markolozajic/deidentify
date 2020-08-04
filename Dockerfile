FROM alpine AS builder

RUN apk add curl
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda3.sh
RUN chmod +x miniconda3.sh

FROM python:3.7

COPY --from=builder miniconda3.sh .
RUN ./miniconda3.sh -b
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
SHELL ["/bin/bash", "-c"]
RUN conda init
RUN conda create -n deidentify python=3.7
RUN echo "conda activate deidentify" > ~/.bashrc
ENV PATH /opt/conda/envs/deidentify/bin:$PATH

RUN conda install -y -c conda-forge spacy=2.3.1
RUN conda install -y -c conda-forge loguru=0.2.5
RUN conda install -y -c conda-forge sklearn-crfsuite=0.3.6
RUN conda install -y -c pytorch pytorch=1.3.0
RUN conda install -y unidecode=1.0.23
RUN conda install -y pandas=0.23.4
RUN pip install flair==0.4.3
RUN python -m spacy download de_core_news_sm

WORKDIR /app/

COPY deidentify/ deidentify/
#RUN python -m deidentify.util.download_model model_bilstmcrf_ons_fast-v0.1.0
COPY demo.py .
COPY data/example_de.txt .

CMD ["bash"]
