FROM python:3.8-slim

WORKDIR /app

COPY . .

# COPY ./checkpoints/train/ckpt-3.data-00000-of-00001 ./checkpoints/train/ckpt-3.data-00000-of-00001

# COPY ./checkpoints/train/ckpt-3.index ./checkpoints/train/ckpt-3.index 

# COPY ./checkpoints/tokenizer.pkl ./checkpoints/tokenizer.pkl

# COPY ./requirements.txt ./requirements.text

# COPY app.py app.py


RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

# CMD ["streamlit","run", "app.py"]
# CMD ["sh", "-c", "streamlit run --server.port $PORT app.py"] 
CMD streamlit run app.py --server.port $PORT