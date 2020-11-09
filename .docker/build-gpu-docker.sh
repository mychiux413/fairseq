docker build -t flashlight:cuda-10.2-base -f .docker/flashlight-CUDA-10.2-Base.Dockerfile . && \
docker build -t wav2letter:cuda-10.2 -f .docker/wav2letter-CUDA-10.2.Dockerfile . && \
docker build -t mychiux413/wav2vec2:cuda-10.2 -f .docker/wav2vec2-CUDA-10.2.Dockerfile . && \
docker push mychiux413/wav2vec2:cuda-10.2