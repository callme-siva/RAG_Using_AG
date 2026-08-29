# 🚀 Deployment & Production Guide

This guide walks you through the steps to deploy your **Streamlit RAG Explorer** application to public cloud platforms (Streamlit Community Cloud, Hugging Face Spaces) and containerized environments (Docker).

---

## 📑 Deployment Options

| Platform | Difficulty | Cost | Best For |
| :--- | :--- | :--- | :--- |
| **Streamlit Community Cloud** | ⭐ Easy (1-Click) | **Free** | Fast public demo, GitHub integration |
| **Hugging Face Spaces** | ⭐ Easy | **Free** | AI community showcase |
| **Docker Container** | ⭐⭐ Intermediate | Self-hosted | Enterprise on-premise, AWS / GCP / Azure |

---

## 🎈 Option 1: Deploy to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud connects directly to your GitHub repository and automatically deploys changes whenever you push to `main`.

### Step-by-Step Instructions:

1. **Sign in to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Click **"Continue with GitHub"** and authenticate.

2. **Create a New App**:
   - Click the **"New app"** button in the top right.
   - Select your repository: `callme-siva/RAG_Using_AG` (or your fork).
   - Set **Branch** to `main`.
   - Set **Main file path** to `app.py`.

3. **Configure API Key Secrets**:
   - Click **"Advanced settings"** at the bottom of the dialog.
   - Under the **Secrets (TOML format)** tab, enter your API keys:
     ```toml
     # Google Gemini API Key (Recommended)
     GOOGLE_API_KEY = "your_actual_gemini_api_key_here"

     # OpenAI API Key (Optional)
     OPENAI_API_KEY = "your_actual_openai_api_key_here"
     ```
   - Click **Save**.

4. **Deploy**:
   - Click **"Deploy!"**.
   - Streamlit will read `requirements.txt`, install dependencies, and launch your live application with a public `https://...streamlit.app` URL.

---

## 🤗 Option 2: Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Set Space SDK to **Streamlit**.
3. Clone the repo and push your files:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push hf main
   ```
4. In your Space **Settings** -> **Variables and secrets**, add `GOOGLE_API_KEY` and `OPENAI_API_KEY` as Secrets.

---

## 🐳 Option 3: Containerized Docker Deployment

If you want to run the application in a production Docker container or on a cloud virtual machine (AWS EC2, Google Cloud Run, Azure App Service):

### 1. Dockerfile
Create a `Dockerfile` in the root directory:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build and Run Container
```bash
# Build the Docker image
docker build -t streamlit-rag-explorer .

# Run container with environment variables
docker run -d -p 8501:8501 \
  -e GOOGLE_API_KEY="your_api_key" \
  --name rag-app streamlit-rag-explorer
```

Access the app at `http://localhost:8501`.

---

## 🔒 Security Best Practices

1. **Never Commit API Keys to Git**:
   - Keep `.env` and `.streamlit/secrets.toml` in your `.gitignore`.
2. **API Key Input in Web UI**:
   - The app allows users to input their own keys in the sidebar without storing them to disk.
3. **Restricting CORS / XSRF**:
   - Streamlit protection flags are already enabled in `.streamlit/config.toml`:
     ```toml
     [server]
     headless = true
     enableCORS = false
     enableXsrfProtection = true
     ```
