import shutil
import kagglehub
from pathlib import Path


def main():
    target_dir = Path("data/external_navoneel")

    # 1. Descargar con KaggleHub
    print("[INFO] Descargando 'navoneel/brain-mri-images-for-brain-tumor-detection'...")
    cached_path = kagglehub.dataset_download(
        "navoneel/brain-mri-images-for-brain-tumor-detection"
    )
    print(f"[INFO] Descargado en caché: {cached_path}")

    # 2. Encontrar la carpeta raíz real (a veces viene anidada como brain_tumor_dataset/brain_tumor_dataset)
    source_path = Path(cached_path)
    if (source_path / "brain_tumor_dataset").exists():
        source_path = source_path / "brain_tumor_dataset"

    # Verificar que contiene 'yes' y 'no'
    if not ((source_path / "yes").exists() and (source_path / "no").exists()):
        print(
            f"[ERROR] La estructura del dataset en {
                source_path
            } no es la esperada (falta carpetas yes/no)."
        )
        return

    # 3. Copiar al directorio del proyecto
    if target_dir.exists():
        print(f"[INFO] Eliminando versión anterior en {target_dir}...")
        shutil.rmtree(target_dir)

    print(f"[INFO] Copiando datos a {target_dir}...")
    shutil.copytree(source_path, target_dir)
    print("[OK] Dataset externo preparado correctamente.")


if __name__ == "__main__":
    main()
