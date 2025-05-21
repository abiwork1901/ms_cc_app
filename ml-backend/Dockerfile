FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create artifacts directory and set permissions
RUN mkdir -p artifacts && chmod 777 artifacts

# Train the model during build
RUN python train.py --outdir artifacts

EXPOSE 8000

# Use uvicorn directly
CMD ["uvicorn", "serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"] 