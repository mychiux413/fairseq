FROM mychiux413/wav2vec2:cpu

RUN git fetch --all && git checkout wav2vec2-server && pip install --editable ./ && fairseq-server --help