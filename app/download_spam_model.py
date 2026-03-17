"""
Download fine-tuned DistilBERT spam detection checkpoint from Google Drive.

Source: github.com/intelliswarm-ai/enterprise-mailbox-assistant/model-fine-tuned-llm
"""
import sys
from pathlib import Path

FOLDER_ID = "1U6bun6U2DgtNej0lc-Suw_sw4EqPLX6W"
MODEL_DIR = Path(__file__).parent.parent / "models" / "spam_detector"
CHECKPOINT_FILE = MODEL_DIR / "checkpoint.pt"


def download_checkpoint():
    """Download checkpoint.pt from Google Drive folder."""
    if CHECKPOINT_FILE.exists():
        size_mb = CHECKPOINT_FILE.stat().st_size / (1024 * 1024)
        print(f"Spam model checkpoint already exists at {CHECKPOINT_FILE} ({size_mb:.1f} MB)")
        return True

    print("Downloading spam detection model checkpoint from Google Drive...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        print("gdown not installed. Run: pip install gdown")
        print(f"Then re-run this script, or manually download checkpoint.pt from:")
        print(f"  https://drive.google.com/drive/folders/{FOLDER_ID}")
        print(f"And place it at: {CHECKPOINT_FILE}")
        return False

    try:
        folder_url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        print(f"Downloading from: {folder_url}")

        gdown.download_folder(
            url=folder_url,
            output=str(MODEL_DIR),
            quiet=False,
            use_cookies=False,
        )

        if CHECKPOINT_FILE.exists():
            size_mb = CHECKPOINT_FILE.stat().st_size / (1024 * 1024)
            print(f"Successfully downloaded checkpoint ({size_mb:.1f} MB)")
            return True

        # Check if downloaded to a subfolder
        checkpoint_files = list(MODEL_DIR.rglob("checkpoint.pt"))
        if checkpoint_files:
            checkpoint_files[0].rename(CHECKPOINT_FILE)
            print(f"Moved checkpoint to {CHECKPOINT_FILE}")
            return True

        print("Error: checkpoint.pt not found after download")
        return False

    except Exception as e:
        print(f"Error downloading model: {e}")
        print(f"\nPlease manually download checkpoint.pt from:")
        print(f"  https://drive.google.com/drive/folders/{FOLDER_ID}")
        print(f"And place it at: {CHECKPOINT_FILE}")
        return False


if __name__ == "__main__":
    success = download_checkpoint()
    sys.exit(0 if success else 1)
