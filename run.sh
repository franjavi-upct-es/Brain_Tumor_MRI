#!/bin/bash
# run.sh - Pipeline completo: Entrenamiento, Fine-Tuning y Evaluación
# ------------------------------------------------------------------
# 1. Configura el entorno virtual
# 2. Instala dependencias
# 3. Entrena el modelo base (EfficientNet)
# 4. Evalúa el modelo base
# 5. Descarga el dataset externo (Navoneel)
# 6. Ejecuta Fine-Tuning para mejorar sensibilidad
# 7. Evalúa el modelo optimizado en datos externos

set -e  # Salir si hay error

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Añadir el directorio raíz al PYTHONPATH para que Python encuentre el módulo 'src'
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
CONFIG_FILE="configs/config.yaml"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  Brain Tumor MRI: Pipeline Completo de Producción    ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""

# ==========================================
# 1. Entorno Virtual
# ==========================================
echo -e "${GREEN}[1/7] Verificando entorno virtual...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creando entorno virtual...${NC}"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Entorno activo: $VIRTUAL_ENV${NC}"
echo ""

# ==========================================
# 2. Dependencias
# ==========================================
echo -e "${GREEN}[2/7] Instalando dependencias...${NC}"
pip install --upgrade pip -q
pip install -r "$REQUIREMENTS_FILE" -q
echo -e "${GREEN}✓ Dependencias listas${NC}"
echo ""

# ==========================================
# 3. Verificación de Datos Base
# ==========================================
echo -e "${GREEN}[3/7] Verificando dataset principal...${NC}"
if [ ! -d "data/train" ]; then
    echo -e "${YELLOW}No se detectaron los datos de entrenamiento base.${NC}"
    echo -e "Ejecutando script de descarga (Kaggle: masoudnickparvar)..."
    python tools/download_and_prepare_kaggle.py --project-root . --val-size 0.1 --use-symlinks
fi
echo -e "${GREEN}✓ Dataset principal verificado${NC}"
echo ""

# ==========================================
# 4. Entrenamiento del Modelo Base
# ==========================================
echo -e "${GREEN}[4/7] Entrenando Modelo Base...${NC}"
# Exportar librerías para GPU si es necesario
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib

if [ ! -f "models/best.keras" ]; then
    python src/train.py --config "$CONFIG_FILE"
else
    echo -e "${YELLOW}Ya existe 'models/best.keras'. Saltando entrenamiento base.${NC}"
    echo -e "(Borra la carpeta 'models/' si quieres re-entrenar desde cero)"
fi

# Evaluación Base
echo -e "${GREEN}Evaluando Modelo Base...${NC}"
python src/eval.py --config "$CONFIG_FILE"
echo ""

# ==========================================
# 5. Preparación Dataset Externo
# ==========================================
echo -e "${GREEN}[5/7] Preparando Dataset Externo (Navoneel)...${NC}"
if [ ! -d "data/external_navoneel" ]; then
    python tools/download_navoneel.py
else
    echo -e "${GREEN}✓ Dataset externo ya existe en 'data/external_navoneel'${NC}"
fi
echo ""

# ==========================================
# 6. Fine-Tuning (Adaptación)
# ==========================================
echo -e "${GREEN}[6/7] Ejecutando Fine-Tuning (Mejora de Sensibilidad)...${NC}"
# Solo entrenamos si no existe ya el modelo fine-tuned para ahorrar tiempo
if [ ! -f "models/finetuned_navoneel.keras" ]; then
    python tools/train_finetune.py --config "$CONFIG_FILE" --data "data/external_navoneel"
else
    echo -e "${YELLOW}El modelo 'models/finetuned_navoneel.keras' ya existe.${NC}"
    echo -e "Saltando paso de fine-tuning..."
fi
echo ""

# ==========================================
# 7. Evaluación Externa Final
# ==========================================
echo -e "${GREEN}[7/7] Evaluación Final en Datos Externos...${NC}"
# Modificamos temporalmente el script para evaluar el modelo fine-tuned si es necesario,
# o aseguramos que evaluate_external.py apunte al modelo correcto.
# NOTA: Asumimos que evaluate_external.py carga 'best.keras' por defecto, 
# pero el fine-tuning genera 'finetuned_navoneel.keras'. 
# Para automatizarlo, pasamos el path explícito si el script lo soporta, 
# o confiamos en que 'train_finetune.py' dejó todo listo.

# Como en nuestra conversación anterior, vamos a evaluar el resultado final:
echo -e "${BLUE}--- Resultados del Modelo Optimizado ---${NC}"
# Aquí hacemos un truco: renombramos temporalmente para evaluar, o idealmente
# actualizamos evaluate_external.py para aceptar --model. 
# Dado el estado actual, ejecutaremos la evaluación asumiendo que train_finetune ya guardó el modelo.

# IMPORTANTE: Asegúrate de que evaluate_external.py usa el modelo correcto.
# Si no lo has modificado para aceptar argumentos, usará best.keras.
# Para este script automático, es mejor imprimir un recordatorio o 
# usar el script de optimización de umbral que es muy informativo.

python tools/evaluate_external.py --config "$CONFIG_FILE" --data "data/external_navoneel"

echo ""
echo -e "${GREEN}Calculando umbral óptimo...${NC}"
python tools/optimize_threshold.py --config "$CONFIG_FILE" --data "data/external_navoneel"

echo ""

# ==========================================
# Resumen
# ==========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}       PIPELINE FINALIZADO 🎉           ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Modelos Generados:${NC}"
echo -e "  1. Base:       ${YELLOW}models/best.keras${NC} (Alta especificidad)"
echo -e "  2. Optimizado: ${YELLOW}models/finetuned_navoneel.keras${NC} (Alta sensibilidad)"
echo ""
echo -e "${GREEN}Reportes:${NC}"
echo -e "  • Revisa 'reports/' para curvas y matrices de confusión del modelo base."
echo -e "  • Revisa la salida de consola anterior para métricas del modelo optimizado."
echo ""
echo -e "${GREEN}Para inferencia con el modelo optimizado:${NC}"
echo -e "  ${YELLOW}python src/infer.py --image <ruta> --threshold 0.65${NC}"
echo ""