# MAGI Ollama Backend (`--ollama`)

> **Opt-in.** Sin `--ollama`, MAGI funciona exactamente como antes (Claude CLI vía
> `claude -p`). Este backend cambia **solo** el motor del *gate de análisis* MAGI;
> la generación de código sigue en Claude Code. Disponible desde **v4.0.0**.

---

## 1. Por qué un backend Ollama para MAGI

MAGI no es generación de código: es **juicio en ensemble** — tres perspectivas
(Melchior/Balthasar/Caspar) que analizan lo mismo y votan por mayoría. Eso cambia
el cálculo costo-vs-token frente al coding:

- **Costo:** un ciclo de MAGI ingiere spec+plan+diff ×3 agentes + razonamiento +
  síntesis (~150–250k tokens), y se itera como gate recurrente. Es el patrón donde
  el costo por token duele. Ollama local ≈ 0 marginal; Ollama Cloud ~$20 plano.
- **Diversidad real:** tres instancias del *mismo* modelo frontier comparten los
  mismos puntos ciegos. Con Ollama se corren **tres modelos open-weight de linajes
  distintos** como los tres magos → diversidad genuina de arquitectura/entrenamiento,
  exactamente el espíritu de MAGI. El voto por mayoría sobre modelos heterogéneos es
  más robusto, y la revisión humana actúa de backstop.

| Tarea | Backend |
|-------|---------|
| Generar código | **Claude Code** (frontier) — sin cambios |
| Gate de análisis MAGI | **Claude** (default) o **Ollama** (`--ollama`, opt-in) |

---

## 2. Cómo usarlo

### Activación
```bash
# Skill:
/magi --ollama
# Orquestador directo:
python skills/magi/scripts/run_magi.py <code-review|design|analysis> <file_or_text> --ollama [--timeout 900]
```
- `--ollama` es **mutuamente excluyente** con `--model` (los modelos por-mago vienen
  de la config, no del CLI). Pasar ambos es error.
- Cada mago usa un **modelo propio** (ver §3).

### Scaffolding de la config
```bash
python skills/magi/scripts/run_magi.py --ollama-init
```
Genera `./.claude/magi-ollama.toml` (no sobrescribe si ya existe) como template
editable, con `base_url` local activo, el trio por defecto, y un header de 2 modos.

### Configuración (capas, merge por-clave)
Precedencia (gana el primero presente): **env > repo `./.claude/magi-ollama.toml` >
global `~/.claude/magi-ollama.toml` > defaults built-in**.

| Clave | Cadena |
|---|---|
| `base_url` | `MAGI_OLLAMA_HOST` → repo → global → `OLLAMA_HOST` → `http://localhost:11434/v1` |
| `api_key` | `MAGI_OLLAMA_API_KEY` → repo → global → `OLLAMA_API_KEY` → *(none)* |
| `models.<mago>` | `MAGI_OLLAMA_MODEL_<MAGO>` → repo `[models]` → global `[models]` → trio default |
| `structured` | `MAGI_OLLAMA_STRUCTURED` → repo → global → `"schema"` |

Ejemplo de `magi-ollama.toml`:
```toml
base_url = "http://localhost:11434/v1"   # OpenAI-compatible base (cualquier path se respeta verbatim; host:port pelado recibe /v1)
# api_key = "sk-..."                       # solo para nube/auth; local no la necesita
[models]
melchior  = "qwen3.5:397b-cloud"
balthasar = "gpt-oss:120b-cloud"
caspar    = "deepseek-v4-pro:cloud"
# structured = "schema"                    # "schema" | "object" | "off"
```
> **Seguridad:** la `api_key` nunca se loguea ni se escribe en artefactos, y se
> redacta en mensajes de error. No commitees un TOML con una key real. Un
> `MAGI_OLLAMA_API_KEY=""` (vacío) significa **explícitamente sin auth** (no hereda
> la del archivo) — útil en CI.

### Dos modos de nube (mismo cliente REST)
- **A) Daemon local + `ollama signin`** *(recomendado)*: `base_url` local, **sin**
  `api_key`. Los modelos `:cloud` corren en la nube de Ollama **sin descargar pesos**
  (solo un manifest); el daemon adjunta tus credenciales. Es el default.
- **B) API cloud directa**: `base_url` del endpoint nube + `api_key`. Para máquinas
  sin daemon local.

### Structured output + fiabilidad
El request usa `response_format` (JSON schema, `strict:false` por portabilidad). Si un
server responde 400 rechazando `response_format`, MAGI hace **un reintento sin él**
(downgrade) y se apoya en el parser+retry existentes. `structured="off"` lo desactiva.

### Preflight (fail-fast)
Antes de lanzar agentes, MAGI verifica que el host responde y que el trio está
disponible. Si falta todo el trio `:cloud` y el daemon no lista ninguno `:cloud`, el
mensaje sugiere `ollama signin`. Un `/models` ausente (404/501) → warn-and-proceed.

---

## 3. Modelos por defecto (lo que envía el plugin)

Trio por defecto = tier **Máximo** (cloud), **tres linajes distintos** a propósito; el
modelo más capaz va a **Caspar** (el crítico es el asiento de mayor apalancamiento —
"losing Caspar biases toward false-positive approval"):

| Mago | Rol | Modelo default | Linaje |
|------|-----|----------------|--------|
| **Melchior** | Científico | `qwen3.5:397b-cloud` | Alibaba |
| **Balthasar** | Pragmático | `gpt-oss:120b-cloud` | OpenAI |
| **Caspar** | Crítico | `deepseek-v4-pro:cloud` | DeepSeek |

> ⚠️ **Recomendaciones de modelos vigentes al 2026-06-07.** El catálogo de Ollama
> cambia con el tiempo; **re-verifica los tags vigentes** en
> [ollama.com](https://ollama.com/search?c=cloud) antes de fijarlos. Los defaults son
> una constante fácilmente actualizable (`DEFAULT_MODELS` en `ollama_config.py`).

### 3 tiers recomendados (al 2026-06-07)

| Tier | Melchior | Balthasar | Caspar | Hardware |
|------|----------|-----------|--------|----------|
| **Ligero** | `qwen3:8b` | `gpt-oss:20b` | `deepseek-r1:8b` | 1 GPU ~16-24 GB (puede serializar) |
| **Balanceado** | `qwen3:32b` | `gpt-oss:20b` | `deepseek-r1:32b` | ~48-64 GB |
| **Máximo (default)** | `qwen3.5:397b-cloud` | `gpt-oss:120b-cloud` | `deepseek-v4-pro:cloud` | Ollama Cloud (`ollama signin`) o 80 GB+ |

Linaje consistente por rol entre tiers: **Caspar = DeepSeek** (el reasoner más fuerte),
**Balthasar = OpenAI** (gpt-oss, esfuerzo ajustable), **Melchior = Qwen** (Alibaba). El tier
Máximo usa tags `:cloud` (sin descarga de pesos, requiere `ollama signin`).

> Nota de VRAM/concurrencia: MAGI lanza los 3 agentes en paralelo; si los modelos no
> coexisten en VRAM, Ollama los serializa. Dimensiona el trio a tu presupuesto, o usa
> el tier Máximo en nube.

---

## 4. Troubleshooting

| Síntoma | Causa probable | Acción |
|---|---|---|
| `No se alcanza Ollama en <host>` | daemon caído / host errado | arranca Ollama; revisa `base_url`/`OLLAMA_HOST` |
| `No :cloud models available ... Run ollama signin` | trio `:cloud` sin sesión nube | `ollama signin` (modo A) o pon `api_key`+base_url nube (modo B) |
| `Missing models on <host>: [...]` | modelos locales no descargados | `ollama pull <model>`, o edita el TOML al tier que tengas |
| `Auth failed (401/403)` | api_key inválida/ausente | corrige `MAGI_OLLAMA_API_KEY` / el TOML |
| 400 + degradado en cada corrida | server no soporta `response_format` | usa `structured="object"` o `"off"` |
| `--model no aplica con --ollama` | pasaste ambos | quita `--model`; los modelos van en el TOML/env |

---

## 5. Referencias

- Implementación: `skills/magi/scripts/ollama_backend.py`, `ollama_config.py`,
  `ollama_preflight.py`, `ollama_init.py`, `backend.py`, `claude_backend.py`,
  `agent_schema.py`.
- Catálogo cloud de Ollama: <https://ollama.com/search?c=cloud>
