class _loc():
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, item):
        return self.obj.loc[item]

class _iloc():
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, item):
        return self.obj.iloc[item]

class _at():
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, item):
        return self.obj.at[item]

class _shape():
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, item):
        return self.obj.shape[item]

class _isin():
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, item):
        return self.obj.isin[item]