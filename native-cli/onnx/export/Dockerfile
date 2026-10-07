# Pinned exporter image for the Gradient ONNX artifact. The toolchain versions
# below reproduce the pinned model.onnx byte-for-byte (see model-assets-linux.json
# "onnx" block); CMake and the installer verify the produced sha256, so any
# drift between the image and the pinned artifact fails loudly.
FROM python:3.14-slim
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 && \
    pip install --no-cache-dir transformers==5.2.0 onnx==1.23.0 safetensors
WORKDIR /export
COPY export.py /export/export.py
ENTRYPOINT ["python", "/export/export.py"]