import shutil
import kagglehub
from pathlib import Path


def main():
    target_dir = Path("data/external_navoneel")

    # 1. Download with KaggleHub
    print("[INFO] Downloading 'navoneel/brain-mri-images-for-brain-tumor-detection'...")
    cached_path = kagglehub.dataset_download(
        "navoneel/brain-mri-images-for-brain-tumor-detection"
    )
    print(f"[INFO] Downloaded to cache: {cached_path}")

    # 2. Find the real root folder (sometimes nested as brain_tumor_dataset/brain_tumor_dataset)
    source_path = Path(cached_path)
    if (source_path / "brain_tumor_dataset").exists():
        source_path = source_path / "brain_tumor_dataset"

    # Verify that it contains 'yes' and 'no'
    if not ((source_path / "yes").exists() and (source_path / "no").exists()):
        print(
            f"[ERROR] Dataset structure at {
                source_path
            } is not as expected (missing yes/no folders)."
        )
        return

    # 3. Copy to project directory
    if target_dir.exists():
        print(f"[INFO] Removing previous version at {target_dir}...")
        shutil.rmtree(target_dir)

    print(f"[INFO] Copying data to {target_dir}...")
    shutil.copytree(source_path, target_dir)
    print("[OK] External dataset prepared successfully.")


if __name__ == "__main__":
    main()
