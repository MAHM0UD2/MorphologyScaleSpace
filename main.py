from pathlib import Path
from src.state import state
from src.feat_gen import load_and_generate

def process_image_folder(input_folder):
    directory = Path(input_folder)

    # Iterate specifically over all BMP files in the folder
    for file_path in directory.glob("*.bmp"):
        print(f"--- Starting Decomposition for: {file_path.name} ---")

        # 1. Delete previous image data
        state.reset()

        # 2. Execute G(f)
        original_h, original_w = load_and_generate(str(file_path))


if __name__ == "__main__":
    process_image_folder(state.input_folder)