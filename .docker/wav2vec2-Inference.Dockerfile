FROM wav2letter/wav2letter:inference-latest

ENV USE_CUDA=0
ENV KENLM_ROOT_DIR=/root/kenlm

# will use Intel MKL for featurization but this may cause dynamic loading conflicts.
ENV USE_MKL=1

ENV LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64:$LD_IBRARY_PATH
WORKDIR /root/wav2letter/bindings/python


RUN pip install --upgrade pip && pip install soundfile packaging scipy flask gevent grpcio && pip install -e .
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /root
RUN git clone https://github.com/mychiux413/fairseq.git
RUN mkdir data

WORKDIR /root/fairseq
RUN git checkout develop
RUN pip install --editable ./ && python examples/speech_recognition/infer.py --help

RUN apt-get update -y && apt-get install -y sox libsox-dev libsox-fmt-all && git clone https://github.com/facebookresearch/WavAugment.git && cd WavAugment && python setup.py develop
