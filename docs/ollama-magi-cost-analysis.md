# Análisis: Ollama vs OpenAI como backend para MAGI

**Fecha:** 2026-06-06
**Contexto:** Decisión de infraestructura — qué backend de modelo usar para el gate de
análisis MAGI (Melchior/Balthasar/Caspar) del flujo SBTDD, optimizando costo vs uso de tokens.
**Alcance:** Aplica solo a **MAGI como juez/gate de análisis**. La generación de código
permanece en Claude Code (modelo frontier).

---

## Conclusión

**Ollama es la mejor opción para MAGI en costo-vs-uso de tokens** y, en este caso particular,
puede incluso **mejorar la calidad del gate**. OpenAI por token solo se justifica si MAGI se
corre muy esporádicamente y se quiere el techo de razonamiento individual más alto — pero en un
gate que se itera varias veces por feature, el costo por token recurrente trabaja en contra.

| Tarea | Backend recomendado |
|-------|---------------------|
| Generación de código | **Claude Code** (frontier) — sin cambios |
| Gate de análisis MAGI | **Ollama** (costo + diversidad de perspectiva) |

---

## Por qué MAGI es un caso distinto

MAGI **no** es generación de código: es una tarea de **juicio/razonamiento en ensemble**
(3 perspectivas independientes + voto por mayoría). Eso cambia por completo el cálculo
costo-vs-token respecto al coding.

### Perfil de tokens (por qué es caro)

Una corrida de MAGI es pesada:

- **3 agentes**, cada uno ingiere spec + plan + diff completo → input ×3.
- Cada agente produce análisis detallado (+ tokens de razonamiento ocultos si es un modelo
  reasoning).
- Síntesis/consenso por encima.
- En el flujo SBTDD es un **gate recurrente**: checkpoint del plan + gate pre-merge, iterando
  hasta 3× si no alcanza el umbral.

Orden de magnitud: **~150–250k tokens por ciclo completo**, y varios ciclos por feature. Es justo
el patrón donde el **costo por token** se nota — lo contrario a un uso esporádico.

---

## Comparación de backends

### OpenAI (API, pago por token)

- **Costo:** variable, escala con uso. Con modelos reasoning clase GPT-5 los **tokens de
  razonamiento se facturan** y son el grueso del gasto. Un gate recurrente y token-heavy puede
  irse a decenas de USD/mes en desarrollo activo.
- **Fuerza:** techo de razonamiento individual alto — cada agente, por sí solo, es más fuerte.
- **Debilidad para MAGI:** se paga premium por algo donde el **ensemble ya compensa** parte de la
  debilidad individual.

### Ollama (local o cloud flat)

- **Costo local:** marginal ≈ 0 (electricidad + hardware propio). Para un gate que se corre muchas
  veces, imbatible.
- **Costo cloud:** ~$20/mes plano, predecible, con límites de uso. Para carga alta y recurrente,
  vence al pago por token.
- **Fuerza clave:** ver "diversidad" abajo.

---

## El argumento decisivo: MAGI premia diversidad, no solo techo

El diseño de MAGI (Melchior/Balthasar/Caspar = tres ángulos distintos) vive de la **diversidad de
perspectivas**. Tres instancias del *mismo* modelo frontier comparten los **mismos puntos ciegos**
— la diversidad es artificial (solo prompts distintos).

Con Ollama se pueden correr **tres modelos open-weight diferentes** como los tres magos:

| Mago | Modelo sugerido | Rol |
|------|-----------------|-----|
| Melchior | DeepSeek-R1 | Razonamiento fuerte |
| Balthasar | gpt-oss-120b | Otra familia de entrenamiento |
| Caspar | Qwen reasoning / QwQ | Tercer ángulo independiente |

Esto da **diversidad genuina de arquitectura/entrenamiento** — exactamente el espíritu de MAGI, y
algo que un solo proveedor cerrado no ofrece. El voto por mayoría sobre modelos heterogéneos es
estadísticamente más robusto que tres clones del mismo modelo.

Los modelos open de razonamiento (clase DeepSeek-R1) ya son **genuinamente buenos** en
crítica/análisis de diseño — que es la tarea de MAGI, no generar código perfecto. Además, la
**revisión humana** actúa como backstop, cubriendo el riesgo de un techo individual algo menor.

---

## Recomendación por situación

| Situación | Mejor opción para MAGI |
|-----------|------------------------|
| GPU local con VRAM suficiente | **Ollama local, 3 modelos distintos** — costo marginal ~0 + diversidad real |
| Sin hardware local, uso alto | **Ollama cloud (~$20 plano)** — predecible, vence al por-token recurrente |
| MAGI muy esporádico + techo máximo | **OpenAI API** — pero se paga premium que el ensemble no necesita |

---

## Riesgos a vigilar al migrar MAGI a Ollama

- **Latencia:** los modelos de razonamiento open son más lentos. Un gate de 3 agentes puede tardar
  bastante en local — aceptable para un gate pre-merge, molesto si se corre muy seguido.
- **VRAM:** cada modelo debe caber **entero** en VRAM. Si hay offload a RAM del sistema, el enlace
  (especialmente en eGPU) estrangula el rendimiento por token. Dimensionar los tres modelos al
  presupuesto de VRAM disponible.
- **Diversidad real vs conveniencia:** correr tres modelos distintos exige tenerlos los tres
  descargados/cargables; correr el mismo modelo 3× es más simple pero pierde la ventaja de
  diversidad que justifica la elección.

---

## Separación de responsabilidades (resumen)

- **Generar código** → Claude Code (frontier). Sin cambios.
- **Gate de análisis MAGI** → Ollama (costo + diversidad de perspectiva).

Son tareas distintas: el techo frontier importa para *producir* código correcto; para *juzgar*
diseño en ensemble, la diversidad + costo bajo de Ollama gana.
