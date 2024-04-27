from sklearn.model_selection import train_test_split
import torch
import numpy
import cairo
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from data import load_drawings, pad_drawings, get_toy_drawings
from model import SketchCritic, SketchDecoder


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", type=str)


class Sketch:
    def __init__(self):
        self.side = 256.0

        line_diameter = 16
        padding = 16

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(self.side), int(self.side))
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_width(line_diameter)

        total_padding = padding * 2.0 + line_diameter
        new_scale = float(self.side) / float(self.side + total_padding)
        self.ctx.scale(new_scale, new_scale)
        self.ctx.translate(total_padding / 2.0, total_padding / 2.0)

        self.ctx.set_source_rgb(0.0, 0.0, 0.0)
        self.ctx.paint()
        self.ctx.set_source_rgb(1.0, 1.0, 1.0)

        self.pen_down = True
        self.pen_position = numpy.array([0.0, 0.0])
        self.done = False


    def add_pred(self, state: numpy.ndarray):
        self.pen_position = numpy.clip(
            self.pen_position + [state[0] * 255, state[1] * 255],
            0.0, 255.0
        )
        arg_max = numpy.argmax(state[2:])

        if self.done:
            return

        if arg_max == 0:
            if not self.pen_down:
                self.ctx.move_to(*self.pen_position)
                self.pen_down = True

            else:
                self.ctx.line_to(*self.pen_position)

        if arg_max == 1:
            self.ctx.stroke()
            self.pen_down = False

        if arg_max == 2:
            self.done = True


    def plot(self):
        if self.pen_down:
            self.ctx.line_to(*self.pen_position)
            self.ctx.stroke()
            self.ctx.move_to(*self.pen_position)

        data = self.surface.get_data()
        raster_image = numpy.copy(numpy.asarray(data, dtype=numpy.float32)[::4])
        raster_image = raster_image / 255.0
        raster_image = raster_image.reshape((int(self.side), int(self.side)))

        plt.imshow(raster_image * 255)
        plt.show()


def batch_seq_to_one_hot(batch_seq_tensor):
    batch_size, seq_length, num_classes = batch_seq_tensor.size()
    max_indices = torch.argmax(batch_seq_tensor, dim=2)
    one_hot = torch.zeros(batch_size, seq_length, num_classes)
    one_hot[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_length).unsqueeze(0), max_indices] = 1.0
    return one_hot


if __name__ == "__main__":
    # load config and args
    args = parser.parse_args()
    config = TrainingConfig()
    model_config = ModelConfig()

    # load model
    model = SketchDecoder(model_config)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device("cpu")))
    model.eval()

    # use criterion for making gmm
    criterion = SketchCritic()

    """ compute loss
    drawings = load_drawings("data/square.ndjson", config.data_sparsity)
    drawings = pad_drawings(drawings, config.max_sequence_length)
    drawings = torch.tensor(drawings, dtype=torch.float32)
    train_drawings, test_drawings = train_test_split(drawings, train_size=0.8)
    train_dataloader = DataLoader(train_drawings, batch_size=1024, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_drawings, batch_size=1024, shuffle=True, drop_last=True)
    
    train_samples = next(iter(train_dataloader))
    test_samples = next(iter(test_dataloader))
    with torch.no_grad():
        train_outputs = model(train_samples)
        test_outputs = model(test_samples)
    #tmp = [element[0, 0].unsqueeze(0) for element in train_outputs]
    #print(tmp); exit()
    train_position_loss, train_pen_loss = criterion(train_samples, *train_outputs)
    test_position_loss, test_pen_loss = criterion(test_samples, *test_outputs)
    print(f"train_position_loss: {train_position_loss}")
    print(f"train_pen_loss: {train_pen_loss}")
    print(f"test_position_loss: {test_position_loss}")
    print(f"test_pen_loss: {test_pen_loss}")
    exit(0)
    """

    #""" draw one
    drawings = load_drawings("data/square.ndjson", 10)
    drawings = pad_drawings(drawings, config.max_sequence_length)
    drawings = torch.tensor(drawings, dtype=torch.float32)
    #drawings = get_toy_drawings(1)

    drawing = drawings[0]

    sketch = Sketch()
    positions = drawing[:, :2]
    positions_next = torch.roll(positions, -1, dims=0)
    delta_positions = positions_next - positions
    delta_positions[drawing[:, 4] == 1.0] = torch.tensor([0.0, 0.0])

    deltas_drawing = drawing.clone()
    deltas_drawing[:, :2] = delta_positions
    sequence = torch.tensor(pad_drawings([[[0, 0, 0, 0, -100]]], config.max_sequence_length), dtype=torch.float32)
    #print(drawing)
    #print(sequence)
    #exit(0)
    sketch2 = Sketch()
    for index, delta_state in enumerate(deltas_drawing):
        with torch.no_grad():
            output = model(sequence)  # [output, batch, seq]

        # unpack output
        next_output = [element[:, index].unsqueeze(0) for element in output]
        next_output[2] *= 1
        next_output[3] *= 1
        print(next_output)
        delta_pred = next_output[:-1]
        pen_pred = next_output[-1]

        # generate pen state
        dist = torch.distributions.categorical.Categorical(probs=pen_pred)
        pen_state = torch.zeros((1, 1, 3))
        pen_state[0, 0, dist.sample()] = 1.0

        # generate delta
        mixture_model = criterion.make_mixture_model(*delta_pred)
        next_delta = mixture_model.sample((1, ))[0]

        # pack into next state
        pred_delta_state = torch.concatenate((next_delta, pen_state), dim=2)[0, 0]

        print((delta_state, pred_delta_state))
        if index <= 4:
            pred_delta_state = delta_state.clone()

        sketch.add_pred(delta_state)
        sketch2.add_pred(pred_delta_state)

        sequence[0, index + 1] = drawing[index + 1]
        sequence[0, index + 1] = torch.concatenate((torch.tensor(numpy.array([[sketch2.pen_position / 255]]), dtype=torch.float32), pen_state), dim=2)
        sketch.plot()
        sketch2.plot()

        if index >= config.max_sequence_length - 2:
            break

    sketch.plot()
    #sketch2.plot()
    exit(0)
    #"""

    # generate predictions
    sketch = Sketch()
    sequence = torch.tensor(pad_drawings([[[0, 0, 0, 0, -100]]], config.max_sequence_length), dtype=torch.float32)
    for index in range(99):
        print("-----")
        # infer next movement
        with torch.no_grad():
            output = model(sequence)  # [output, batch, seq]
        
        # only use output of next token in sequence
        next_output = [element[:, index].unsqueeze(0) for element in output]
        print(next_output[0])
        print(next_output[1])
        print(next_output[2])
        print(next_output[3])

        # unpack output
        delta_pred = next_output[:-1]
        pen_pred = next_output[-1]

        # generate pen state
        dist = torch.distributions.categorical.Categorical(probs=pen_pred)
        pen_state = torch.zeros((1, 1, 3))
        pen_state[0, 0, dist.sample()] = 1.0

        # generate delta
        mixture_model = criterion.make_mixture_model(*delta_pred)
        next_delta = mixture_model.sample((1, ))[0]
        print((next_delta, pen_pred))

        # pack into next state
        pred = torch.concatenate((next_delta, pen_state), dim=2)
        sketch.add_pred(pred[0, 0].numpy())
        state = torch.concatenate((  # don't @ me
            torch.tensor(numpy.array([[sketch.pen_position / 255]]), dtype=torch.float32),
            pen_state
        ), dim=2)
        print(state)

        # add to sequence
        sequence[0, index + 1] = state
        sketch.plot()

        # stop if requested
        if pen_state[0, 0, 2] == 1.0:
            break

    sketch.plot()
