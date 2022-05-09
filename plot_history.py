from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from util.CsvReader import CsvReader


def main(input_path,output_path,output_name, npz_hist: dict, csv_hist: dict):
    npz_input_paths = [(Path(input_path).joinpath(h, "history.npz"), n) for n, h in npz_hist.items()]
    csv_input_paths = [(Path(input_path).joinpath(h, "history_watched.csv"), n) for n, h in csv_hist.items()]
    datum = [(np.array(np.load(str(input_path), allow_pickle=True)["history"], ndmin=1)[0], n)
             for input_path, n in npz_input_paths]
    datum.extend([(CsvReader(input_path).to_dict(), n) for input_path, n in csv_input_paths])
    metrics = ["val_loss", "val_PSNR"]

    for metric in metrics:
        for data, setting_name in datum:
            plt.plot(list(map(float,data[metric])), label=setting_name)
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(Path(output_path).joinpath(output_name+'_'+metric + ".png"))
        plt.cla()


if __name__ == '__main__':
    main("","","", dict(), {'test': "."})
