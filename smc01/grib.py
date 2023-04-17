"""Our compatibility layer for GRIB, when the available tools don't do the trick for us."""

import weakref

import eccodes


class GribMessage:
    """Thin wrapper to make to eccodes interface to a GRIB message more pythonic.
    
    You should not create instances of this class yourself. They should be created
    by the GribFile class."""

    def __init__(self, msg_id):
        self.msg_id = msg_id
        self.__finalizer = weakref.finalize(self, eccodes.codes_release, msg_id)

    def __getitem__(self, key):
        if eccodes.codes_is_defined(self.msg_id, key):
            return eccodes.codes_get(self.msg_id, key)
        else:
            raise KeyError(f"Key {key} is missing.")

    def __str__(self):
        return f"{self['name']} {self['level']} {self['typeOfLevel']}"


class GribFile:
    """Thin wrapper to make the eccodes interface more pythonic."""

    def __init__(self, path, mode="rb"):
        self.path = path
        self.mode = mode
        self.handle = None

    def __enter__(self):
        self.handle = open(self.path, self.mode)
        return self

    def __exit__(self, *args):
        self.handle.close()

    def __iter__(self):
        self.handle.seek(0)
        return self

    def __next__(self):
        msg_id = eccodes.codes_grib_new_from_file(self.handle)

        if msg_id:
            return GribMessage(msg_id)
        else:
            raise StopIteration

    def write(self, msg: GribMessage):
        eccodes.codes_write(msg.msg_id, self.handle)
