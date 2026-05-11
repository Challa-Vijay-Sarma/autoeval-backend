# Backend-only Docker image. The frontend ships as a separate service
# (see frontend/Dockerfile).
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY golden_trajectory.py failure_trajectory.py run_all.py ./
COPY system_prompt.md knowledge_base.md ./
COPY platform_app ./platform_app
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn platform_app.main:app --host 0.0.0.0 --port ${PORT}"]
