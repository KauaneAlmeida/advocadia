FROM python:3.11-slim

# Instalar dependências do sistema só uma vez (cacheia bem)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório da app
WORKDIR /app

# Criar usuário não-root
RUN adduser --disabled-password --gecos '' appuser

# Copiar requirements e instalar (essa parte você pode sobrescrever depois sem rebuild total)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

# Copiar aplicação
COPY app/ ./app/

# Dar permissão
RUN chown -R appuser:appuser /app

# Trocar para usuário não-root
USER appuser

# Expor porta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Iniciar FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

