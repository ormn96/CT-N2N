from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from util.CsvReader import CsvReader

def main(input_path,npz_hist,csv_hist):
    npz_input_paths = [Path(input_path).joinpath(h,"history.npz") for h in npz_hist]
    csv_input_paths = [Path(input_path).joinpath(h, "history_watched.csv") for h in csv_hist]
    datum = [(np.array(np.load(str(input_path), allow_pickle=True)["history"], ndmin=1)[0], input_path.parent.name)
             for input_path in npz_input_paths]
    datum.extend([(CsvReader(input_path).to_dict(), input_path.parent.name) for input_path in csv_input_paths])
    metrics = ["val_loss", "val_PSNR"]

    for metric in metrics:
        for data, setting_name in datum:
            plt.plot(data[metric], label=setting_name)
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(metric + ".png")
        plt.cla()


if __name__ == '__main__':
    main("",[],["."])