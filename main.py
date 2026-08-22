from pathlib import Path
from src.state import state
from PIL import Image as PILImage
import numpy as np
from src.utils.image import Image
from src.pipeline.feat_gen import feature_generator
from src.pipeline.step_func import compression
from src.pipeline.reconstruction import reconstruct


def process_image_folder(input_folder):
    directory = Path(input_folder)

    # Iterate specifically over all BMP files in the folder
    for file_path in directory.glob("*.bmp"):
        print(f"--- Starting Decomposition for: {file_path.name} ---")

        # Delete previous image data
        state.reset()

        raw_img = PILImage.open(str(file_path))
        pixel_matrix = np.array(raw_img, dtype=int)
        f = Image(pixel_matrix)

        # 1. Execute G(f)
        feature_generator(f)

        t_arr = state.F_1["t"].values.astype(int)
        v_arr = state.F_1["v"].values.astype(int)
        active_mask = np.ones(len(state.F_1), dtype=bool)

        # 2. Step function
        # Todo
        # Todo: Remove this as soon as you find a better approach
        for i in range(100):
            compression(t_arr, v_arr, active_mask)

        state.F_1["v"] = v_arr
        state.F_1 = state.F_1[active_mask].reset_index(drop=True)

        # 3. Reconstruction
        reconstructed_u = reconstruct()

        # Saving output
        path = Path(state.output_folder) / f"{file_path.name}"
        reconstructed_u.save_bmp(path)

        diff_image = f.absolute_log_difference(reconstructed_u)
        diff_image.save_bmp(Path(state.output_folder) / f"Diff {file_path.name}")

        print(f"    Compressed: {state.ell / state.L}%")

        print(f"--- Ending Pipeline for: {file_path.name} ---")

if __name__ == "__main__":
    process_image_folder(state.input_folder)