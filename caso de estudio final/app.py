import base64
import csv
import io
import json
import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

EXTRACTION_FIELDS = """
Extrae la siguiente informacion del articulo cientifico. Responde SOLO con un JSON valido, sin texto adicional ni backticks:

{
  "titulo": "titulo completo del articulo",
  "autores": "apellidos de los autores separados por coma",
  "anio": "ano de publicacion",
  "revista": "nombre de la revista o journal",
  "pregunta_investigacion": "pregunta o problema de investigacion principal (1-2 oraciones)",
  "objetivo": "objetivo principal del estudio (1 oracion)",
  "poblacion": "poblacion o muestra estudiada (tipo, tamano si se menciona)",
  "diseno": "diseno metodologico (ej: ensayo clinico, estudio de cohorte, revision sistematica...)",
  "intervencion": "intervencion o exposicion evaluada (si aplica)",
  "desenlace_principal": "desenlace o resultado principal medido",
  "resultados_clave": "hallazgos principales con datos concretos (2-3 oraciones)",
  "conclusion": "conclusion principal de los autores (1-2 oraciones)",
  "limitaciones": "limitaciones del estudio mencionadas por los autores",
  "nivel_evidencia": "nivel de evidencia estimado (I: meta-analisis/ECA, II: cohortes, III: caso-control, IV: series de casos, V: opinion experto)",
  "palabras_clave": "3-5 palabras clave del articulo"
}
"""

# Instrucciones base — se usan en todos los providers
CHAT_SYSTEM_PROMPT = """Eres un asistente especializado en analisis de literatura cientifica medica.
REGLAS ESTRICTAS que debes seguir siempre:
1. Responde SIEMPRE en espanol, sin excepciones.
2. Cita explicitamente de que articulo o documento viene cada afirmacion (usa el nombre del archivo, autores o ano).
3. Si tienes multiples documentos, analiza y compara TODOS, no solo el ultimo.
4. Si hay contradicciones entre estudios, senalalas.
5. Si la pregunta no puede responderse con los documentos disponibles, dilo claramente.
6. Usa lenguaje cientifico apropiado."""


def extract_text_from_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def build_text_document_context(documents, cache=None):
    """
    Construye el bloque de texto con todos los documentos.
    Si se pasa cache (dict nombre->texto), lo usa en vez de re-extraer.
    """
    chunks = []
    for doc in documents:
        name = doc["name"]
        if cache is not None and name in cache:
            text = cache[name]
        else:
            text = extract_text_from_pdf(doc["bytes"])
            if not text.strip():
                text = "[No se pudo extraer texto legible del PDF]"
            text = text[:18000]
            if cache is not None:
                cache[name] = text
        chunks.append(f"=== INICIO DOCUMENTO: {name} ===\n{text}\n=== FIN DOCUMENTO: {name} ===")
    return "\n\n".join(chunks)


class BaseLLMProvider:
    def generate_from_documents(self, prompt, documents, max_tokens):
        raise NotImplementedError

    def extract_json(self, pdf_bytes, filename):
        raw = self.generate_from_documents(
            prompt=EXTRACTION_FIELDS,
            documents=[{"name": filename, "bytes": pdf_bytes}],
            max_tokens=1500,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)

    def build_chat_prompt(self, question, history, protocol_ctx=""):
        """Construye el bloque de pregunta + historial + protocolo."""
        lines = []
        if protocol_ctx:
            lines.append(protocol_ctx)
            lines.append("")
        if history:
            lines.append("--- HISTORIAL DE CONVERSACION ---")
            for msg in history:
                role = "Usuario" if msg.get("role") == "user" else "Asistente"
                lines.append(f"{role}: {msg.get('content', '')}")
            lines.append("--- FIN HISTORIAL ---\n")
        lines.append(f"Pregunta del usuario: {question}")
        lines.append("\nIMPORTANTE: Responde en espanol. Cita el nombre del documento para cada afirmacion.")
        if protocol_ctx:
            lines.append("Ten en cuenta el protocolo de revision sistematica definido arriba al responder.")
        return "\n".join(lines)

    def answer_question(self, question, history, documents, protocol_ctx=""):
        chat_block = self.build_chat_prompt(question, history, protocol_ctx)
        return self.generate_from_documents(
            prompt=chat_block,
            documents=documents,
            max_tokens=2000,
        )


class AnthropicProvider(BaseLLMProvider):
    def __init__(self):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Falta ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

    def generate_from_documents(self, prompt, documents, max_tokens):
        content = []
        for doc in documents:
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.standard_b64encode(doc["bytes"]).decode("utf-8"),
                },
                "title": doc["name"],
            })
        content.append({"type": "text", "text": prompt})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=CHAT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text


class GoogleProvider(BaseLLMProvider):
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Falta GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            os.getenv("LLM_MODEL", "gemini-1.5-flash"),
            system_instruction=CHAT_SYSTEM_PROMPT,
        )

    def generate_from_documents(self, prompt, documents, max_tokens):
        parts = []
        for doc in documents:
            parts.append({"mime_type": "application/pdf", "data": doc["bytes"]})
        parts.append(prompt)
        response = self.model.generate_content(
            parts, generation_config={"max_output_tokens": max_tokens}
        )
        return response.text


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LIGHTNING_API_KEY")
        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY o LIGHTNING_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LIGHTNING_BASE_URL")
        if not base_url:
            raise RuntimeError("Falta OPENAI_BASE_URL o LIGHTNING_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv("LLM_MODEL", "lightning-ai/gemma-4-31B-it")

    def generate_from_documents(self, prompt, documents, max_tokens):
        doc_context = build_text_document_context(documents) if documents else "No hay documentos."
        # Sistema + docs primero, pregunta al final (mejor atención en modelos openai-compatible)
        final_prompt = f"{CHAT_SYSTEM_PROMPT}\n\n{doc_context}\n\n{prompt}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [{"type": "text", "text": final_prompt}]}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not base.endswith("/api/generate"):
            base = base.rstrip("/") + "/api/generate"
        self.url = base
        self.model = os.getenv("LLM_MODEL", "gpt-oss:20b")
        self._doc_cache = {}  # nombre → texto, persiste entre turnos

    def _call_ollama(self, prompt, max_tokens):
        """Llamada directa al API de Ollama."""
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=(10, 300),
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
        return response.json()["response"]

    def generate_from_documents(self, prompt, documents, max_tokens):
        # Actualizar caché con docs nuevos
        if documents:
            for doc in documents:
                name = doc["name"]
                if name not in self._doc_cache:
                    text = extract_text_from_pdf(doc["bytes"])
                    self._doc_cache[name] = text[:18000] if text.strip() else "[Sin texto legible]"

        if not self._doc_cache:
            return self._call_ollama(
                f"{CHAT_SYSTEM_PROMPT}\n\nNo hay documentos adjuntos.\n\n{prompt}",
                max_tokens
            )

        # ── ESTRATEGIA MAP-REDUCE ──────────────────────────────────────────
        # Modelos locales tienen ventana de contexto limitada y olvidan
        # documentos anteriores cuando el prompt es muy largo.
        # Solucion: consultar cada doc por separado (map) y consolidar (reduce).

        print(f"[ollama] Ejecutando map-reduce sobre {len(self._doc_cache)} documentos...")

        # MAP: extraer respuesta relevante de cada documento individualmente
        partial_answers = []
        for name, text in self._doc_cache.items():
            map_prompt = (
                f"Eres un asistente de literatura cientifica medica. Responde SIEMPRE en espanol.\n\n"
                f"=== DOCUMENTO: {name} ===\n{text}\n=== FIN DOCUMENTO ===\n\n"
                f"Basandote SOLO en el documento anterior, responde de forma concisa:\n"
                f"{prompt}\n\n"
                f"Si el documento no contiene informacion relevante, escribe exactamente: "
                f"'Sin informacion relevante en este documento.'\n"
                f"Menciona el nombre del documento al inicio de tu respuesta."
            )
            print(f"[ollama] MAP -> {name}")
            try:
                partial = self._call_ollama(map_prompt, max_tokens=600)
                partial_answers.append(f"[{name}]:\n{partial.strip()}")
            except Exception as e:
                partial_answers.append(f"[{name}]: Error al procesar: {e}")

        # REDUCE: consolidar todas las respuestas en una sola
        reduce_prompt = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            f"Se consultaron {len(partial_answers)} articulos cientificos. "
            f"Respuestas individuales por articulo:\n\n"
            f"{'='*60}\n"
            + "\n\n".join(partial_answers) +
            f"\n{'='*60}\n\n"
            f"Sintetiza todo en una respuesta UNICA, completa y estructurada en espanol. "
            f"Cita de que articulo viene cada dato. "
            f"Si distintos articulos usan diferentes metricas o metodos, comparalos. "
            f"Ignora las secciones que digan 'Sin informacion relevante'.\n\n"
            f"Pregunta original: {prompt}"
        )

        print(f"[ollama] REDUCE -> consolidando {len(partial_answers)} respuestas")
        return self._call_ollama(reduce_prompt, max_tokens)

    def clear_cache(self):
        self._doc_cache = {}


def build_provider():
    provider_name = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if provider_name == "anthropic":
        return AnthropicProvider()
    if provider_name == "google":
        return GoogleProvider()
    if provider_name in {"openai_compatible", "lightning"}:
        return OpenAICompatibleProvider()
    if provider_name == "ollama":
        return OllamaProvider()
    raise RuntimeError("LLM_PROVIDER debe ser 'anthropic', 'google', 'openai_compatible' o 'ollama'")


llm = build_provider()


def decode_docs(docs_b64):
    return [{"name": d["name"], "bytes": base64.b64decode(d["b64"])} for d in docs_b64]


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/extract", methods=["POST"])
def extract():
    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"error": "No se enviaron archivos"}), 400

    results, errors = [], []
    for f in files:
        filename = f.filename
        try:
            pdf_data = f.read()
            data = llm.extract_json(pdf_data, filename)
            data["_archivo"] = filename
            results.append({"filename": filename, "data": data, "ok": True})
        except json.JSONDecodeError:
            errors.append({"filename": filename, "error": "No se pudo parsear la respuesta como JSON"})
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})

    return jsonify({"results": results, "errors": errors})


@app.route("/chat", methods=["POST"])
def chat():
    import traceback
    body = request.get_json() or {}
    question = body.get("question", "")
    history = body.get("history", [])
    docs_b64 = body.get("docs", [])
    protocol_ctx = body.get("protocol", "").strip()

    if not question:
        return jsonify({"error": "Pregunta vacia"}), 400

    cache_keys = list(llm._doc_cache.keys()) if hasattr(llm, "_doc_cache") else []
    print(f"\n[chat] Pregunta: {question[:80]}")
    print(f"[chat] _doc_cache ({len(cache_keys)} docs): {cache_keys}")
    print(f"[chat] Protocolo: {'Si' if protocol_ctx else 'No'}")

    try:
        documents = decode_docs(docs_b64)
        answer = llm.answer_question(question, history, documents, protocol_ctx)
        return jsonify({"answer": answer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/apply_protocol", methods=["POST"])
def apply_protocol():
    """Evalua si un articulo cumple el protocolo de inclusion/exclusion."""
    import traceback
    body = request.get_json() or {}
    article = body.get("article", "")
    protocol = body.get("protocol", "")

    if not article or not protocol:
        return jsonify({"error": "Faltan datos"}), 400

    prompt = (
        "Eres un experto en revision sistematica de literatura medica.\n\n"
        f"{protocol}\n\n"
        "Evalua el siguiente articulo y determina si CUMPLE los criterios de inclusion "
        "y NO incumple ningun criterio de exclusion.\n\n"
        f"ARTICULO:\n{article}\n\n"
        "Responde SOLO con JSON valido sin backticks:\n"
        '{"decision": "include" o "exclude", "reason": "explicacion breve en espanol de max 2 oraciones"}'
    )

    try:
        raw = llm.generate_from_documents(prompt=prompt, documents=[], max_tokens=200)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"decision": "pending", "reason": str(e)}), 500


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    if hasattr(llm, "clear_cache"):
        llm.clear_cache()
    return jsonify({"ok": True})


@app.route("/cache_status", methods=["GET"])
def cache_status():
    """Devuelve los nombres de documentos que el backend ya tiene en caché."""
    cached = list(llm._doc_cache.keys()) if hasattr(llm, "_doc_cache") else None
    return jsonify({"cached": cached})


@app.route("/preload", methods=["POST"])
def preload():
    """
    Recibe todos los PDFs como archivos multipart, los extrae y los guarda en cache.
    Llamar desde el frontend al hacer 'Usar en chat'.
    """
    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"error": "No se enviaron archivos"}), 400

    loaded = []
    errors = []

    for f in files:
        name = f.filename
        try:
            pdf_bytes = f.read()
            if not pdf_bytes:
                errors.append({"name": name, "error": "Archivo vacio"})
                continue
            text = extract_text_from_pdf(pdf_bytes)
            if not text.strip():
                text = "[No se pudo extraer texto legible del PDF]"
            if hasattr(llm, "_doc_cache"):
                llm._doc_cache[name] = text[:18000]
            loaded.append(name)
        except Exception as e:
            errors.append({"name": name, "error": str(e)})

    print(f"[preload] Cargados: {loaded}")
    print(f"[preload] Errores: {errors}")
    print(f"[preload] Caché ahora tiene: {list(llm._doc_cache.keys()) if hasattr(llm, '_doc_cache') else 'N/A'}")
    return jsonify({"loaded": loaded, "errors": errors, "total": len(loaded)})


@app.route("/export_csv", methods=["POST"])
def export_csv():
    body = request.get_json() or {}
    rows = body.get("rows", [])
    if not rows:
        return jsonify({"error": "Sin datos"}), 400

    fields = [
        "_archivo", "titulo", "autores", "anio", "revista",
        "pregunta_investigacion", "objetivo", "poblacion", "diseno",
        "intervencion", "desenlace_principal", "resultados_clave",
        "conclusion", "limitaciones", "nivel_evidencia", "palabras_clave",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    return jsonify({"csv": output.getvalue()})


if __name__ == "__main__":
    print("Thesis Tool corriendo en http://localhost:5000")
    print(f"Proveedor LLM : {os.getenv('LLM_PROVIDER', 'anthropic')}")
    print(f"Modelo        : {os.getenv('LLM_MODEL', '(default del proveedor)')}")
    app.run(debug=True, port=5000)