FROM python:3.13.3

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 7860
EXPOSE 7860


CMD ["uvicorn", "main:app", "--host","0.0.0.0", "--port","7860"]

