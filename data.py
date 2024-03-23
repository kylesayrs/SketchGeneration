from typing import List

import json
import tqdm


def load_drawings(file_path: str, sparsity: int = 1) -> List[List[int]]:
    drawings = []

    with open(file_path, "r") as file:
        lines = file.readlines()

        for line in tqdm.tqdm(lines[::sparsity], desc="Loading data"):
            entry = json.loads(line)
            if not entry["recognized"]:
                continue
                
            drawing = [[0, 0, 0, 1, 0]]  # start with pen lifted
            for stroke_index, (stroke_xs, stroke_ys) in enumerate(entry["drawing"]):
                positions = list(zip(stroke_xs, stroke_ys))

                # do normal movement
                for x, y in positions:
                    drawing.append([
                        x / 255,
                        y / 255,
                        1, 0, 0
                    ])

                # modify end stroke
                drawing[-1][3] = 0
                drawing[-1][4] = 1

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
