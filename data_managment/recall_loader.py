import glob
import os

import pandas as pd

from const import RCL_DATA_PATTERN


class RecallLoader:

    # Load every recall CSV and retain its source filename
    # ----------------------------------------------------
    def load(self):
        recall_paths = sorted(glob.glob(RCL_DATA_PATTERN))
        if not recall_paths:
            raise FileNotFoundError(
                f"No recall files match the pattern: {RCL_DATA_PATTERN}"
            )

        recall_frames = []
        for path in recall_paths:
            recall_data = pd.read_csv(path, dtype="string")
            recall_data["source_file"] = os.path.basename(path)
            recall_frames.append(recall_data)

        return pd.concat(recall_frames, ignore_index=True)
