from torch.utils.data import Dataset
import os
from typing import Union, List, Tuple, Callable


class ImageDataset(Dataset):
    def __init__(self,
                 path: str,
                 recursive: bool = True,
                 transform: Callable = None,
                 suffixes: Union[Tuple[str], str] = None
                 ):
        self.item_paths = []
        self.root_path = path
        self.transform = transform

        # Set check if extension is valid
        if suffixes and isinstance(suffixes, str):
            suffixes = (suffixes, )
            check_ext: Callable[[str], bool] = lambda x: x.lower() in suffixes
        else:
            check_ext: Callable[[str], bool] = lambda x: True

        def walk(current_path):
            # Check all files in directory
            for file in os.listdir(current_path):
                # Create filepath
                filepath = os.path.join(current_path, file)
                # Check if it is a file
                if os.path.isfile(filepath):
                    # Fetch extension
                    ext = os.path.splitext(file)[1]
                    # If the extension is valid add path to list
                    if check_ext(ext):
                        # Add to file paths
                        self.item_paths.append(filepath)
                # Check if it is a directory
                elif recursive and os.path.isdir(filepath):
                    # If it is a directory, walk the directory
                    walk(filepath)

        walk(path)

    def __len__(self):
        return len(self.item_paths)

    def __getitem__(self, idx):
        filepath = os.path.join(self.root_path, self.item_paths[idx])



        return None
