class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.

    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.

    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.
    """


class TsFileParseException(Exception):
    """Should be raised when parsing a .ts file and the format is incorrect."""


class ArffFileParseException(Exception):
    """Should be raised when parsing a .arff file and the format is incorrect."""


class LongFormatDataParseException(Exception):
    """Should be raised when parsing a .csv file with long-formatted data."""
