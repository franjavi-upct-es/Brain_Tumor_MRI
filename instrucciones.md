A partir de los _artefactos que has subido_ (JSONs + dashboards) el problema principal no es “que el modelo no aprenda”, sino que el pipeline está **tomando decisiones clínicas (Tumor/Healthy) con una lógica demasiado rígida y poco consistente entre scripts**, y además **estás mezclando comparativas que no están alineadas** (umbral fijo vs TTA vs calibración), lo cual te está ocultando la causa real de los fallos.

## 1) Qué está pasando realmente (lectura técnica de tus resultados)

En el dataset externo (179 imágenes: **92 sanas** y **87 con tumor**, deducido de tus matrices), el patrón es claro:

- **Base**: muy conservador. Casi no “dispara alarmas” (FP=1), pero **se deja muchos tumores** (FN=47).
- **Fine-tuned**: mejora fuerte en sensibilidad (FN baja a 21), pero **paga** con bastantes falsos positivos (FP=20).
- **Focal (+TTA)**: vuelve a ser “cero falsas alarmas” (FP=0), pero **pierde** sensibilidad frente a fine-tuning (FN=39).

Esto no es raro: estás moviéndote en el trade-off típico **sensibilidad vs especificidad**. El error es que tu pipeline no está controlando ese trade-off explícitamente (con umbral/objetivo/criterio), y en algunos puntos lo está **forzando** a 0.5 y listo.

## 2) Dónde te estás equivocando (fallos estructurales del pipeline)

### A) Umbral “clínico” hardcodeado e inconsistente (P0)

Tu `config.yaml` define `inference.threshold: 0.5`.
Pero, peor: en `adaptive_retrain.py` el binarizado está **hardcodeado a 0.5** (`pred_binary = 1 if prob_tumor > 0.5 else 0`).

Esto significa:

- La mejora de sensibilidad **no la estás “gobernando”** con un criterio clínico (p.ej. “FN rate < 0.15”) sino con un umbral fijo.
- Tu `optimize_threshold.py` puede recomendar un umbral, pero **no se integra automáticamente** en la inferencia/evaluación posterior.

### B) Estás optimizando F1 cuando tu objetivo declarado es clínico (P0)

`optimize_threshold.py` busca el mejor umbral por **F1**.
Sin embargo, tu propio “criterio clínico” (en `adaptive_retrain.py`) marca como objetivo algo tipo recall > 0.85 y FN rate < 0.15.

F1 **no** garantiza ese constraint. Si tu prioridad es _no perder tumores_, el tuning debería ser:

- “Maximiza recall sujeto a FP<=X” o
- “Minimiza FN sujeto a especificidad mínima” o
- coste asimétrico explícito.

### C) Comparativas “apple-to-orange”: TTA/calibración no están alineadas (P0)

En `config.yaml` tienes `inference.tta: false` pero en focal has evaluado con `tta_enabled: true, tta_samples: 5`.
Así, el gráfico comparativo no representa _solo_ el efecto de la loss, sino una mezcla de:

- arquitectura + pesos
- (quizá) calibración
- (en focal) TTA
- umbral fijo

### D) Calidad de datos / preprocessing: no estás usando el control de calidad (P0)

Tu configuración permite QC pero lo tienes desactivado (`quality_filter: false`) aunque guardas metadatos (`save_metadata: true`).

Y en tu “gallery of misclassified” aparecen casos que parecen **frames “vacíos/negros”** o claramente anómalos. Eso normalmente indica:

- fallo de skull-stripping / máscara,
- crop erróneo,
- normalización que colapsa intensidades,
- o incluso lectura/decodificación defectuosa.

Si esos casos entran a evaluación, te van a generar **FN de alta confianza** (y tu galería sugiere precisamente eso).

### E) Estás infrautilizando losses pensadas para “no perder tumores” (P1)

Tienes implementado `TverskyLoss` con `beta > alpha` explícitamente orientado a penalizar FN (missed tumors).
Y también mencionas Weighted BCE orientado a asimetría FN/FP.

Sin embargo, tu “improvement_report” se centra en Focal + label smoothing + augmentation + TTA.
Focal no siempre es la mejor herramienta cuando el objetivo es **recall a toda costa** (a menudo funciona mejor un coste asimétrico directo).

### F) Estás infrautilizando Grad-CAM para auditar por qué fallas (P1)

Ya tienes utilidades para Grad-CAM y guardado por batch.
Pero el flujo actual parece más centrado en métricas globales que en “por qué estos FN ocurren”. Sin explicación local, es fácil “tocar hyperparams” sin arreglar el fallo real (preproc, dominio, slice sin tumor visible, etc.).

## 3) Lista de tareas priorizada (clara, ejecutable, por módulos)

### P0 — Bloqueantes (si no haces esto, seguirás “disparando a ciegas”)

1. **Unificar la fuente de verdad para Tumor/Healthy**
   - Definir una única función `tumor_score()` (idealmente en logits) y usarla en: `evaluate_external`, `infer`, `adaptive_retrain`, `optimize_threshold`.
   - Mantener el criterio actual “Tumor = 1 - P(no_tumor)” pero sin duplicaciones ni variantes. (Hoy ya aparece así en `optimize_threshold.py` y `adaptive_retrain.py`).

2. **Eliminar umbrales hardcodeados**
   - Sustituir `> 0.5` por `cfg.inference.threshold` en _todos_ los scripts (especialmente `adaptive_retrain.py`).

3. **Cambiar el criterio de “threshold optimization” a criterio clínico**
   - Reemplazar “best F1” por una política del tipo:
     - “elige el menor umbral que cumpla especificidad ≥ S_min” o
     - “elige el umbral que minimice FN con FP≤K”.

   - (F1 es una métrica estadística, no una regla clínica).

4. **Alinear comparativas (sin mezclar TTA/calibración)**
   - Ejecutar 3 evaluaciones comparables **con el mismo set de toggles**:
     - TTA off/on (igual para todos)
     - calibración on/off (igual para todos)
     - umbral fijo vs umbral optimizado (igual para todos)

   - Ahora mismo focal no es comparable con base/finetuned porque incluye TTA explícito en el resultado.

5. **Activar control de calidad o excluir casos patológicos**
   - Activar `quality_filter` o, como mínimo, generar un reporte de “imágenes con máscara vacía / intensidad colapsada / área cerebral mínima” y excluirlas o tratarlas con fallback.

---

### P1 — Correcciones de rendimiento (para atacar FN sin destrozar todo)

6. **Usar pérdidas orientadas a FN (no solo Focal)**
   - Probar `TverskyLoss(beta>alpha)` o `WeightedBinaryCrossEntropy` en la fase que persigue sensibilidad.

7. **Auditoría sistemática de FN con Grad-CAM**
   - Para los top-N FN (por confianza), guardar:
     - imagen original + preprocesada
     - Grad-CAM de `no_tumor` y del “tumor_score”
     - metadatos de preproc

   - Esto te dirá si el modelo mira “fuera del cerebro”, si el preproc borró el tumor o si el tumor es imperceptible en esa slice.

8. **Revisar augmentations “tumor-focused” por realismo**
   - En tu reporte hablas de rotaciones agresivas (±54°). Eso suele ser demasiado para RM clínica (riesgo de generar distribuciones irreales y FPs).
   - Restringir TTA/augment a transformaciones físicamente plausibles.

9. **Separar “validación externa” de “adaptación”**
   - Si vas a fine-tunear en Navoneel, crea un split **hold-out** (no tocarlo durante fine-tuning) para que la cifra sea interpretable.

---

### P2 — Robustez y producto (cuando P0/P1 estén controlados)

10. **Ensembling y decisión por riesgo**

- En vez de un solo modelo/umbral, usar un “regla de seguridad”:
  - si el modelo es incierto, marcar como “revisión” (triage) en lugar de “Healthy”.

- Especialmente útil si tu objetivo es reducir FN.

11. **Reporte y dashboards: trazabilidad completa**

- Cada métrica debe guardar: modelo, commit/config, preprocessing mode, threshold, TTA, calibration(T), dataset hash, conteos TN/FP/FN/TP.
- Tu `run.sh` hoy ejecuta evaluación y luego optimización de umbral; esto invita a confusión si no queda registrado qué umbral se usó dónde.

## 4) Archivos adicionales que necesito para afinar el diagnóstico (si me los pasas, te devuelvo plan aún más quirúrgico)

Sin hacerte perder tiempo: para confirmar _exactamente_ dónde se generan los “negros”, y si hay fuga de datos en fine-tuning, necesito ver:

- `tools/evaluate_external.py` completo (cómo aplica `use_calibration`, cómo carga T, cómo preprocesa).
- `tools/train_finetune.py` (si hace split train/val/test o entrena con todo).
- El `error_log.csv` (aunque sea un extracto de 20 filas) para cruzar ruta → preproc → pred → confianza.

Con lo anterior puedo decirte, con precisión, si el cuello de botella es **preprocesado**, **umbral**, **calibración**, **label noise** o **estrategia de adaptación**, y darte tareas P0/P1 ya mapeadas a funciones concretas.
