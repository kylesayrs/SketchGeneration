import matplotlib.pyplot as plt

from data import load_drawings


if __name__ == "__main__":
    drawings = load_drawings("data/flip flops.ndjson")
    
    sequence_lengths = [len(drawing) for drawing in drawings]

    #plt.yscale("log")
    plt.hist(sequence_lengths)
    plt.show()
