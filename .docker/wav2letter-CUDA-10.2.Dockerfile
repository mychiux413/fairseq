# ==================================================================
# module list
# ------------------------------------------------------------------
# flashlight       master       (git, CUDA backend)
# ==================================================================

FROM flashlight:cuda-10.2-base

RUN cd /root && git clone https://github.com/facebookresearch/wav2letter.git && \
    cd /root/wav2letter && git checkout v0.2

# ==================================================================
# flashlight https://github.com/facebookresearch/flashlight.git
# ------------------------------------------------------------------
RUN cd /root && git clone --recursive https://github.com/facebookresearch/flashlight.git && \
    cd /root/flashlight && git checkout v0.2 && mkdir -p build && \
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CUDA && \
    make -j$(nproc) && make install && \
# ==================================================================
# wav2letter with GPU backend
# ------------------------------------------------------------------
    export MKLROOT=/opt/intel/mkl && export KENLM_ROOT_DIR=/root/kenlm && \
    cd /root/wav2letter && mkdir -p build && \
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DW2L_LIBRARIES_USE_CUDA=ON -DW2L_BUILD_INFERENCE=ON && \
    make -j$(nproc) 