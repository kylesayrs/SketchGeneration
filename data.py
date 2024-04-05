from typing import List, Union

import json
import tqdm
import torch


def load_drawings(file_path: str, sparsity: int = 1) -> List[List[int]]:
    drawings = []

    with open(file_path, "r") as file:
        lines = file.readlines()

        for line in tqdm.tqdm(lines[::sparsity], desc="Loading data"):
            entry = json.loads(line)
            if not entry["recognized"]:
                continue
                
            drawing = [[0, 0, 0, 1, 0]]  # start with pen lifted
            for _stroke_index, (stroke_xs, stroke_ys) in enumerate(entry["drawing"]):
                positions = list(zip(stroke_xs, stroke_ys))

                # do normal movement
                for x, y in positions:
                    drawing.append([
                        x / 255,
                        y / 255,
                        1, 0, 0
                    ])

                # modify end stroke
                drawing[-1][2] = 0
                drawing[-1][3] = 1

            # end drawing
            drawing.append([0, 0, 0, 0, 1])
            drawings.append(drawing)

    return drawings


def pad_drawings(drawings: List[List[int]], max_sequence_length: int) -> List[List[int]]:
    for index in tqdm.tqdm(range(len(drawings)), desc="Padding data"):
        sequence_length = len(drawings[index])
        num_samples_to_add = max_sequence_length - sequence_length

        if num_samples_to_add > 0:
            drawings[index].extend([[0, 0, 0, 0, 1]] * num_samples_to_add)

        if num_samples_to_add < 0:
            drawings[index] = drawings[index][:max_sequence_length]

        assert len(drawings[index]) == max_sequence_length

    return drawings


class DrawingsDataset(torch.utils.data.Dataset):
    def __init__(self, drawings, scale_factor: Union[float, None] = 0.2):
        self.drawings = drawings
        self.scale_factor = scale_factor


    def __len__(self) -> int:
        return len(self.drawings)


    def __getitem__(self, index: int) -> torch.Tensor:
        drawing = self.drawings[index]

        if self.scale_factor is not None:
            scale_factor = (  # lerp
                torch.rand(1)[0] * ((1 + self.scale_factor) - (1 - self.scale_factor))
                + (1 - self.scale_factor)
            )
            drawing[:, :2] *= scale_factor

        return drawing
