from typing import List
import os


class GetFilesInFolder:
    def __init__(self) -> None:
        pass

    def get_name_files(self, folder_path: str) -> List[str]:
        files = os.listdir(folder_path)
        files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
        return files
