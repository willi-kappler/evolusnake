

# Python std lib:
import json
import math
import sys

# External libs:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():
    with open(sys.argv[1], "r") as f:
        data = json.load(f)

    positions = data["positions"]
    length: float = 0.0

    (x0, y0) = positions[-1]
    for (x1, y1) in positions:
        length += math.hypot(x0 - x1, y0 - y1)
        (x0, y0) = (x1, y1)

    print(f"{length=}")

    (fig, axes) = plt.subplots()
    poly = plt.Polygon(positions, ec="k", fc="none")
    axes.add_patch(poly)
    axes.relim()
    axes.autoscale_view()
    plt.show()


if __name__ == "__main__":
    main()
