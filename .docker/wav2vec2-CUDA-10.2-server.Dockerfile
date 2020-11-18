FROM mychiux413/wav2vec2:cuda-10.2

RUN git fetch --all && git checkout wav2vec2-server && pip install soundfile packaging scipy flask gevent grpcio && pip install --editable ./ && fairseq-server --help