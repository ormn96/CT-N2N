import csv


class _CsvRow:
    __slots__ = ['_raw_row', '_header']

    def __init__(self, row, header):
        self._raw_row = row
        self._header = header

    def __getattr__(self, item):
        if self._header is None:
            return super().__getattribute__(item)
        try:
            return self._raw_row[self._header.index(item)]
        except ValueError:
            return super().__getattribute__(item)

    def __getitem__(self, item):
        if isinstance(item, str) and self._header is not None:
            return self._raw_row[self._header.index(item)]
        return self._raw_row[item]

    def __repr__(self):
        return self._raw_row.__repr__()


class CsvReader:
    __slots__ = ['_raw_data', '_has_header', '_header']

    def __init__(self, file_name, has_header=True):
        self._has_header = has_header
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            if has_header:
                self._header = reader.__next__()
            else:
                self._header = None
            data = [row for row in reader]
            self._raw_data = data
            # self._raw_data = np.array(data)

    def __getitem__(self, item):
        return _CsvRow(self._raw_data[item], self._header)

    def __iter__(self):
        header = self._header
        data = self._raw_data

        class CsvIter:
            __slots__ = ['_iter']

            def __init__(self):
                self._iter = data.__iter__()

            def __next__(self):
                return _CsvRow(self._iter.__next__(), header)

        return CsvIter()

    def to_dict(self):
        if not self._has_header:
            raise NotImplementedError("dict keys missing")
        d = dict()
        for k in self._header:
            d[k] = []
        for v in self:
            for k in self._header:
                d[k].append(v[k])

        return d
