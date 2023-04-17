"""Utilities to select messages from a GRIB message stream. The general interface for an
extractor is a callable that returns a label if we keep the variable, and False if we
don't keep it."""


def from_short_name(message):
    to_extract = [
        "al",
        "hpbl",
        "prate",
        "prmsl",
        "thick",
        "10si",
        "10wdir",
        "10u",
        "10v",
        "2d",
        "2r",
        "2t",
    ]
    if message.shortName in to_extract:
        return message.shortName
    else:
        return False


class ShortNameLevelExtractor:
    """Use both the short name and the level to decide if we keep the message"""

    def __init__(self, name, levels):
        self.name = name
        self.levels = levels

    def __call__(self, message):
        if message.shortName == self.name:
            for level in self.levels:
                if message.level == level:
                    return f"{self.name}_{level}"
        return False


class CompositeExtractor:
    def __init__(self, extractors):
        self.extractors = extractors

    def __call__(self, message):
        for e in self.extractors:
            is_included = e(message)
            if is_included:
                return is_included
        return False


DEFAULT_EXTRACTOR = CompositeExtractor(
    [
        from_short_name,
        ShortNameLevelExtractor("t", [850, 500]),
        ShortNameLevelExtractor("gh", [1000, 850, 500]),
        ShortNameLevelExtractor("q", [850, 500]),
        ShortNameLevelExtractor("u", [500]),
        ShortNameLevelExtractor("v", [500]),
    ]
)
