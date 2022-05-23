from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from util.CsvReader import CsvReader


def main(input_path, output_path, output_name, npz_hist: List[tuple], csv_hist: List[tuple], *, legend_prop=None,
         legend_pos='best',metrics=None):
    if metrics is None:
        metrics = ["val_loss", "val_PSNR"]
    npz_input_paths = [(Path(input_path).joinpath(h, "history.npz"), n) for n, h in npz_hist]
    csv_input_paths = [(Path(input_path).joinpath(h, "history_watched.csv"), n) for n, h in csv_hist]
    datum = [(np.array(np.load(str(input_path), allow_pickle=True)["history"], ndmin=1)[0], n)
             for input_path, n in npz_input_paths]
    datum.extend([(CsvReader(input_path).to_dict(), n) for input_path, n in csv_input_paths])

    for metric in metrics:
        for data, setting_name in datum:
            plt.plot(list(map(float, data[metric])), label=setting_name)
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.legend(loc=legend_pos, prop=legend_prop)
        plt.savefig(Path(output_path).joinpath(output_name + '_' + metric + ".png"))
        plt.cla()


if __name__ == '__main__':
    main("", "", "", [], [('test', ".")])
