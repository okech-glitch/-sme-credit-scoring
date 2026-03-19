# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies as root (required for compiling LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to comply with Hugging Face Spaces security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy dependencies
COPY --chown=user requirements-api.txt .

# Install python dependencies strictly for inference
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the API server, frontend, and specifically the extracted ML models
COPY --chown=user app.py .
COPY --chown=user index.html .
COPY --chown=user ["Financial Health Data/api_models/", "./Financial Health Data/api_models/"]

# Hugging Face explicitly requires Port 7860
EXPOSE 7860

# Command to run the FastAPI server on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
