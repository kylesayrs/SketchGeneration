import torch
import numpy
import cairo
import argparse
import matplotlib.pyplot as plt

from config import ModelConfig, TrainingConfig
from model import SketchCritic, SketchDecoder


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", type=str)


"""
def strokes_to_raster(
    strokes: List[List[List[int]]],
    side: int = 50,
    line_diameter: int = 16,
    padding: int = 16
) -> numpy.ndarray:
    if len(strokes) <= 0:
        return numpy.zeros((1, side, side), dtype=numpy.float32)

    original_side = 256.0

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2.0 + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2.0, total_padding / 2.0)

    # don't offset to center, not necessary (as of now)

    # clear background
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.paint()

    # draw strokes, this is the most cpu-intensive part
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    for stroke in strokes:
        ctx.move_to(stroke[0][0], stroke[1][0])
        # skip first because we've already moved to it
        for i in range(1, len(stroke[0])):
            ctx.line_to(stroke[0][i], stroke[1][i])
        ctx.stroke()

    data = surface.get_data()
    raster_image = numpy.copy(numpy.asarray(data, dtype=numpy.float32)[::4])
    raster_image = raster_image / 255.0
    raster_image = raster_image.reshape((side, side))
    raster_image = numpy.expand_dims(raster_image, axis=0)  # one channel image

    return raster_image
"""


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

        self.pen_down = False
        self.pen_position = numpy.array([0.0, 0.0])
        self.done = False


    def add_state(self, state: numpy.ndarray):
        self.pen_position += numpy.clip(state[:2] * 255, 0.0, 255.0)
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
        self.ctx.stroke()

        data = self.surface.get_data()
        raster_image = numpy.copy(numpy.asarray(data, dtype=numpy.float32)[::4])
        raster_image = raster_image / 255.0
        raster_image = raster_image.reshape((int(self.side), int(self.side)))

        plt.imshow(raster_image * 255)
        plt.show()


if __name__ == "__main__":
    # load config and args
    args = parser.parse_args()
    config = TrainingConfig()
    model_config = ModelConfig()

    # load model
    decoder = SketchDecoder(model_config)
    decoder.load_state_dict(torch.load(args.checkpoint_path))
    decoder.eval()

    # use criterion for making gmm
    criterion = SketchCritic()

    # TODO: feed in first stroke from existing data

    # generate predictions
    sketch = Sketch()
    state = torch.tensor([[[0, 0, 1, 0, 0]]], dtype=torch.float32)
    for index in range(config.max_sequence_length):
        # infer next movement
        with torch.no_grad():
            output, hidden_state = decoder(state)
        
        # unpack output
        position_pred = output[:-1]
        pen_pred = output[-1]

        # generate position
        mixture_model = criterion.make_mixture_model(*position_pred)
        next_position = mixture_model.sample((1, ))[0]
        
        if numpy.argmax(pen_pred.numpy()) == 2:
            break

        state = torch.concatenate((next_position, pen_pred), dim=2)
        sketch.add_state(state[0, 0].numpy())
        print(state)

    sketch.plot()
