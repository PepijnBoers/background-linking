FROM adoptopenjdk/openjdk11

RUN apt-get update && \
    apt-get install python3.6 -y && \
    apt update && \
    apt install git -y && \
    apt install python3-pip -y && \
    mkdir /opt/background-linking && \
    cd /opt/ && \
    git clone https://github.com/castorini/anserini-tools.git && \
    cd anserini-tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../.. && \
    cd anserini-tools/eval && cd ndeval && make && cd ../../..
    
COPY . /opt/background-linking

RUN cd /opt/background-linking/ && \
    pip3 install -e .

WORKDIR "/opt/background-linking/bglinking"

ENTRYPOINT ["python3", "-u", "reranker.py"]