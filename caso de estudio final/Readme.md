# Thesis Tool · Analisis de Literatura Medica

App web local para extraer informacion estructurada de articulos cientificos en PDF y hacer preguntas sobre ellos usando un proveedor LLM configurable.

## Instalacion

### 1. Requisitos
- Python 3.9+
- Una API key valida del proveedor que vayas a usar

### 2. Instalar dependencias

```bash
pip install -r Requirements.txt
```

### 3. Configurar `.env`

Para Anthropic:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-4-20250514
```

Para Google:

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=tu_api_key
LLM_MODEL=gemini-1.5-flash
```

Para Lightning u otro endpoint OpenAI-compatible:

```env
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://lightning.ai/api/v1/
OPENAI_API_KEY=c35b33e5-aa57-45b9-82ea-f28e35fd7f77
LLM_MODEL=lightning-ai/gemma-4-31B-it
```

Para Ollama local:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1/
OLLAMA_API_KEY=ollama
LLM_MODEL=qwen3:4b
```

Notas:
- `LLM_PROVIDER` puede ser `anthropic`, `google` o `openai_compatible`.
- Tambien puede ser `ollama`.
- `LLM_MODEL` es opcional. Si no lo defines, la app usa un default por proveedor.
- Una API key de Google no funciona con Anthropic, y una de Anthropic no funciona con Google.
- Una API key de Lightning no funciona con Google Generative AI. Necesita `OPENAI_BASE_URL`.
- Para Ollama, normalmente solo necesitas tener el servidor local corriendo y el modelo descargado.

### 4. Correr la app

```bash
python app.py
```

Abre tu navegador en: `http://localhost:5000`

## Uso

### Modo Extraccion
1. Arrastra tus PDFs al area de carga.
2. Haz clic en `Extraer informacion`.
3. Exporta los resultados como CSV si lo necesitas.
4. Haz clic en `Usar en chat` para pasar los PDFs al modo conversacional.

### Modo Chat
- Los PDFs se reenvian en cada turno para mantener el contexto.
- Puedes comparar estudios, resumir resultados, revisar limitaciones y buscar contradicciones entre articulos.

## Nota sobre PDFs en proveedores OpenAI-compatible

Para Anthropic y Google la app envia los PDFs como documentos cuando el SDK lo soporta. En proveedores OpenAI-compatible como Lightning, la app extrae el texto del PDF localmente y lo envia como contexto al modelo.

Con `ollama` ocurre lo mismo: el texto del PDF se extrae localmente y luego se envia al modelo como contexto.
