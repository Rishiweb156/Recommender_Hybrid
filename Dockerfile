# Dockerfile

# --- Stage 1: Builder ---
# This stage installs all dependencies, including build tools.
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker layer caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Image ---
# This stage creates the final, lean image for production.
FROM python:3.10-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY . .

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port Streamlit will run on
EXPOSE 8501

# The command to run the Streamlit app
# Healthcheck ensures the container is running properly
HEALTHCHECK CMD streamlit hello
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]