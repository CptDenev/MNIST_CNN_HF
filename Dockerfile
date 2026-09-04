FROM python:3.10-slim

WORKDIR /app

# no need for CUDA as the model size is 1,2 Mo
RUN pip install --no-cache-dir \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY mnist_cnn_final.pth .

EXPOSE 7860

CMD ["python", "app.py"]