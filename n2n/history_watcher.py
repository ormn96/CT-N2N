from pathlib import Path
import csv
import tensorflow.keras as keras


class HistoryWatcher(keras.callbacks.Callback):

    def __init__(self, *, output_path: str):
        """
        create history in the end of every epoch

        :param output_path:
            path to folder to write the object after each epoch
        """
        super().__init__()
        if not output_path.endswith(".csv"):
            raise ValueError(f"file must be an csv file")
        self._output_path = output_path

    def _check_file(self):
        return not Path(self._output_path).exists()

    def on_epoch_end(self, epoch, logs={}):
        first = self._check_file()
        with open(self._output_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, dialect='excel', fieldnames=sorted(logs.keys()))
            if first:
                writer.writeheader()
            writer.writerow(logs)
