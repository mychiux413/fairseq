FROM wav2letter:cuda-10.2

ENV USE_CUDA=1
ENV KENLM_ROOT_DIR=/root/kenlm

# will use Intel MKL for featurization but this may cause dynamic loading conflicts.
# ENV USE_MKL=1

ENV LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64:$LD_IBRARY_PATH
WORKDIR /root/wav2letter/bindings/python

RUN pip install --upgrade pip && pip install tensorboardX soundfile packaging pyarrow && pip install -e .

WORKDIR /root
ENV FAIRSEQ_BRANCH=develop
RUN git clone https://github.com/mychiux413/fairseq.git && echo "uncache 201106-3"
RUN git clone https://github.com/NVIDIA/apex
RUN mkdir data

WORKDIR /root/fairseq
RUN git checkout $FAIRSEQ_BRANCH && pip install --editable ./ && python examples/speech_recognition/infer.py --help

WORKDIR /root/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./ && python -c "import apex"


WORKDIR /root
RUN apt-get update -y && apt-get install -y sox libsox-dev libsox-fmt-all && git clone https://github.com/facebookresearch/WavAugment.git && cd WavAugment && python setup.py develop

WORKDIR /root/fairseq