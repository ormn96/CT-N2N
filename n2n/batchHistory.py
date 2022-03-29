import tensorflow.keras as keras


class BatchHistory(keras.callbacks.Callback):
    """
    create history in the end of every batch

        Parameters
        ----------
            metrics :
                list of values to log.

                default ['loss'].

            output_path :
                path to folder to write the object after the train ended, if None, do not save automatically.

                the function save_history(self, output_path="") will be called
                with this parameter at the end of the training.

                default = None
        Notes
        -----
            use the following code to access the saved history:
                np.load(str(output_path+'batch_history.npz'), allow_pickle=True)['batch_history']
    """

    def __init__(self, *, metrics=None, reduced=True, output_path=None):
        super().__init__()
        if metrics is None:
            metrics = ['loss']
        self.history = None
        self._metrics = metrics
        self._output_path = output_path
        if not reduced:
            def on_epoch_begin(epoch, logs={}):
                for k in self.metrics:
                    self.history[k].append([])
            self.on_epoch_begin = on_epoch_begin

            def on_train_batch_end(batch, logs={}):
                for k in self.metrics:
                    self.history[k][-1].append(logs.get(k))
            self.on_train_batch_end = on_train_batch_end

    def on_train_begin(self, logs={}):
        self.history = {k: [] for k in self._metrics}

    def on_train_batch_end(self, batch, logs={}):
        for k in self._metrics:
            self.history[k].append(logs.get(k))

    def on_train_end(self, logs={}):
        if self._output_path:
            self.save_history(self._output_path)

    def save_history(self, output_path=""):
        import os
        import numpy as np
        from pathlib import Path
        os.makedirs(name=output_path, exist_ok=True)
        np.savez(str(Path(output_path).joinpath("batch_history.npz")), batch_history=self.history)
