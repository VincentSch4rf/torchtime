from __future__ import annotations

import re
import warnings
from enum import Enum, auto
from typing import Dict, List, Union, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from ..exceptions import TsFileParseException
from ..utils import stack_pad


class TsTagValuePattern(Enum):
    """
    Enumeration holding the known `.ts` file headers tag value types in the form of regex expressions.
    """
    BOOLEAN = re.compile(r"(?:tru|fals)e")
    ANY_CONNECTED_STRING = re.compile(r"\w+")
    INTEGER_NUMBER = re.compile(r"\d+")
    CLASS_LABEL = re.compile(r"(?:tru|fals)e(?:(?<=true)((?: [^\s]+)+)|(?<=false))")
    # ((?:tru|fals)e)(?(?<=true)((?: \w+)+))(?=\s)


class TsTag(str, Enum):
    """
    Enumeration holding the names of the known `.ts` file tag names.
    """
    PROBLEM_NAME = "problemName"
    TIMESTAMPS = "timeStamps"
    MISSING = "missing"
    EQUAL_LENGTH = "equalLength"
    SERIES_LENGTH = "seriesLength"
    CLASS_LABEL = "classLabel"
    UNIVARIATE = "univariate"
    DIMENSIONS = "dimensions"


class TSFileLoader:
    """
    File loader that can load time series files in sktimes `.ts` file format.

    Args:
            filepath (str): The path to the `.ts` file.
            nan_replace_value (int, float or str, optional): The value, by which the missing value indicator "?"
                should be replaced. Default: "NaN".
    """

    class State(Enum):
        """
        TSFileLoader's internal parsing state.
        """
        PREFACE = 0
        HEADER = 1
        BODY = 2
        BODY_TIME_STAMPS = 21

    # Dict mapping known `.ts` file header tags to their respective parsing expression
    header_info: Dict[TsTag, TsTagValuePattern] = {
        TsTag.PROBLEM_NAME: TsTagValuePattern.ANY_CONNECTED_STRING,
        TsTag.TIMESTAMPS: TsTagValuePattern.BOOLEAN,
        TsTag.MISSING: TsTagValuePattern.BOOLEAN,
        TsTag.EQUAL_LENGTH: TsTagValuePattern.BOOLEAN,
        TsTag.SERIES_LENGTH: TsTagValuePattern.INTEGER_NUMBER,
        TsTag.CLASS_LABEL: TsTagValuePattern.CLASS_LABEL,
        TsTag.UNIVARIATE: TsTagValuePattern.BOOLEAN,
        TsTag.DIMENSIONS: TsTagValuePattern.INTEGER_NUMBER
    }

    required_meta_info: List[TsTag] = [TsTag.PROBLEM_NAME, TsTag.CLASS_LABEL, TsTag.EQUAL_LENGTH, TsTag.MISSING,
                                       TsTag.TIMESTAMPS]

    def as_tensor(self, return_targets: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """Return the loaded data as a 3-dimensional tensor of the form (N, C, S).

        Keyword Args:
            return_targets (bool):
        Returns:
            torch.Tensor: A 3 dimensional tensor.
        """
        data_ = []
        if len(self.data) == 0:
            self.parse()
        for dim in self.data:
            data_.append(stack_pad(dim))
        data_ = torch.permute(torch.stack(data_, dim=-1), (0, 2, 1))
        if self.header[TsTag.CLASS_LABEL] and return_targets:
            return data_, self.targets
        return data_

    def as_dataframe(self, return_targets: bool = False) -> pd.DataFrame:
        """Return the loaded data as a pandas dataframe.

        Keyword Args:
            return_targets (bool): Identifies whether the targets should be included in the returned dataframe. If
            True, the targets will be added as an additional column 'targets' to the dataframe. This only has an effect
            if there are class labels available in the datafile that was parsed in the first place.
        Returns:
            pd.DataFrame: A nested pandas dataframe holding the dimensions as columns and the number examples as rows,
            where every cell contains a pandas Series containing a univariate time series.
            If `return_targets` is set, it will also contain a column 'targets' that contains the class labels of every
            example.
        """
        data = pd.DataFrame(dtype=np.float32)
        if len(self.data) == 0:
            self.parse()
        for dim in range(0, len(self.data)):
            data["dim_" + str(dim)] = self.data[dim]
        if self.header[TsTag.CLASS_LABEL] and return_targets:
            data["targets"] = self.targets
        return data

    def get_classes(self):
        """Return the classes found in the '.ts' file

        Returns:
            List[str]: List of class names as string.
        """
        if self.header[TsTag.CLASS_LABEL]:
            return self.header[TsTag.CLASS_LABEL]
        else:
            raise AttributeError(f"The '.ts' file {self.filename} does not have any class labels")

    def __init__(self, filepath: str, nan_replace_value: Union[int, float, str] = "NaN"):
        self.filename = filepath
        self.line_number = 1
        self.file = open(filepath, "r", encoding="utf-8")
        self.state = self.State.PREFACE
        self.header = {k: None for k in self.header_info.keys()}
        self.data = []
        self.targets = []
        self.dim = None
        self.series_length = 0
        self.timestamp_type = None
        self.nan_replace_value = nan_replace_value

    def parse_header(self, line: str) -> None:
        """Parses a line of a `.ts` file header and updates the internal state of the loader with the extracted
        information.

        Args:
            line (str): The header line to parse.

        Returns:
            None
        """
        if not line.startswith("@"):
            raise TsFileParseException(
                f"Line number {self.line_number} was interpreted as HEADER but does not start with '@'!"
            )
        line = line[1:]
        if len(line) == 0:
            raise TsFileParseException(
                f"Line number {self.line_number} contains an empty tag!"
            )

        tokens = [token.strip() for token in line.split(" ", maxsplit=1)]
        token_len = len(tokens)

        if token_len == 1:
            raise TsFileParseException(
                f"tag '{tokens[0]}' at line number {self.line_number} requires an associated value!"
            )
        tag = TsTag(tokens[0])
        value_pattern = self.header_info[tag]
        value = value_pattern.value.match(tokens[1])
        if value:
            if len(value.groups()) > 1:
                raise TsFileParseException(
                    "Value extractor should return exactly ONE match!"
                )
            if len(value.groups()) > 0:
                value = value.group(1)
            else:
                value = value.group(0)
            self.header[tag] = self.parse_header_value(value, value_pattern)

    def parse_header_value(self, value: str, value_type: TsTagValuePattern) -> Union[bool, str, int, List[str]]:
        """Parse a single header value that was extracted by the header line parser and return its value as the
        appropriate python object.

        Args:
            value (str): Extracted header value that should be parsed.
            value_type (TsTagValuePattern): The expected type of the value, which should be applied.

        Returns:
            bool: If the value is of type BOOLEAN. `value` converted to bool
            str: If the value is of type ANY_CONNECTED_STRING. Returns the stripped value string.
            List[str]: If the value is of type CLASS_LABEL. Returns a list of space separated string class labels.
        """
        if value_type == TsTagValuePattern.BOOLEAN:
            return value == "true"
        if value_type == TsTagValuePattern.ANY_CONNECTED_STRING:
            return value.strip()
        if value_type == TsTagValuePattern.CLASS_LABEL:
            if value is None:
                return False
            return value.strip().split(" ")
        if value_type == TsTagValuePattern.INTEGER_NUMBER:
            try:
                value = int(value)
            except ValueError:
                raise TsFileParseException(
                    f"Value '{value}' in line {self.line_number} could not be interpreted as int"
                )
            return value

    def parse_body(self, line: str) -> None:
        """Parse a line of the `@data` content of a `.ts` file if `@timeStamps` is `False`.

        Args:
            line (str): The `@data` line to parse.

        Returns:
            None
        """
        dimensions = line.split(":")

        if not self.data:
            if not self.header[TsTag.DIMENSIONS]:
                warnings.warn("Meta information for '@dimensions' is missing. Inferring from data.",
                              UserWarning,
                              stacklevel=2)
                self.dim = len(dimensions)
                # last dimension is the target
                if self.header[TsTag.CLASS_LABEL]:
                    self.dim -= 1
            self.data = [[] for _ in range(self.dim)]

        # Check dimensionality of the data of the current line
        # All dimensions should be included for all series, even if they are empty
        line_dim = len(dimensions)
        if self.header[TsTag.CLASS_LABEL]:
            line_dim -= 1
        if line_dim != self.dim:
            raise TsFileParseException(
                f"Inconsistent number of dimensions. Expecting {self.dim} but got {line_dim} "
                f"in line number {self.line_number}."
            )
        # Process the data for each dimension
        for dim in range(0, self.dim):
            dimension = dimensions[dim].strip()
            if dimension:
                dimension = dimension.replace("?", self.nan_replace_value)
                data_series = dimension.split(",")
                data_series = [float(i) for i in data_series]
                dim_len = len(data_series)
                if self.series_length < dim_len:
                    if not self.header[TsTag.EQUAL_LENGTH]:
                        self.series_length = dim_len
                    else:
                        raise TsFileParseException(
                            f"Series length was given as {self.series_length} but dimension {dim} in line "
                            f"{self.line_number} is of length {dim_len}"
                        )
                self.data[dim].append(pd.Series(data_series))
            else:
                self.data[dim].append(pd.Series(dtype="object"))
        if self.header[TsTag.CLASS_LABEL]:
            self.targets.append(dimensions[self.dim].strip())

    def parse_timestamp_tuple(self, tuple_data: str, line_dim: int) -> Tuple[Union[str, int], float]:
        """Parse a timestamp tuple of the form `(<timestamp>,<float>)`

        Args:
            tuple_data (str): The timestamp tuple as string.
            line_dim (str): The dimension of the line being current parsed.

        Returns:
            Tuple[int, float]: If the timestamp can be interpreted as an integer, return a tuple of the timestamp as int
            and the value as float.
            Tuple[str, float]: Else, the timestamp is probably a date, return a tuple of the timestamp as str and the
            value as float. The timestamp is later converted to an actual DateIndex using pandas.
        """
        last_comma_index = tuple_data.rfind(",")
        if last_comma_index == -1:
            raise TsFileParseException(
                f"Dimension {line_dim} on line {self.line_number} contains a tuple that has no comma "
                f"inside of it"
            )
        try:
            value = tuple_data[last_comma_index + 1:]
            value = float(value)
        except ValueError:
            raise TsFileParseException(
                f"Dimension {line_dim} on line {self.line_number} contains a tuple that does not have "
                f"a valid numeric value"
            )

        timestamp = tuple_data[0:last_comma_index]
        try:
            timestamp = int(timestamp)
            line_timestamp_type = int
        except ValueError:
            try:
                timestamp = timestamp.strip()
                line_timestamp_type = str
            except ValueError:
                raise TsFileParseException(
                    f"Dimension {line_dim} on line {self.line_number} contains a tuple that has an invalid "
                    f"timestamp '{timestamp}'"
                )
        if self.timestamp_type is None:
            self.timestamp_type = line_timestamp_type
        if line_timestamp_type != self.timestamp_type:
            raise TsFileParseException(
                f"Dimension {line_dim} on line {self.line_number} contains tuples where the "
                "timestamp format is inconsistent"
            )
        return timestamp, value

    def parse_body_timestamps(self, line: str) -> None:
        """Parse a line of the `@data` content of a `.ts` file if `@timeStamps` is `True`.

        Args:
            line (str): The `@data` line to parse.

        Returns:
            None
        """
        line_len = len(line)
        line_state = None
        char_num = 0
        line_dim = 0
        tuple_data = None
        target = None

        timestamps_for_dim = []
        values_for_dim = []

        if line.startswith("#"):
            raise TsFileParseException(
                f"Comments are not allowed in '@data' in line {self.line_number}"
            )

        while char_num < line_len:
            token = line[char_num]

            if line_state is None and not token.isspace():
                line_state = TsTimestampLineState.START_LINE
            try:
                line_state = line_state.next(token)
            except ValueError as ve:
                raise TsFileParseException(
                    str(ve) + f" in dimension {line_dim} of line {self.line_number}"
                )
            if line_state == TsTimestampLineState.START_TUPLE:
                tuple_data = ""
            elif line_state == TsTimestampLineState.IN_TUPLE:
                tuple_data += token
            elif line_state == TsTimestampLineState.IN_DIM:
                timestamp, value = self.parse_timestamp_tuple(tuple_data, line_dim)
                timestamps_for_dim += [timestamp]
                values_for_dim += [value]
            elif line_state == TsTimestampLineState.IN_DATA:
                if len(self.data) < line_dim + 1:
                    self.data.append([])
                if self.timestamp_type == str:
                    timestamps_for_dim = pd.DatetimeIndex(timestamps_for_dim)
                self.data[line_dim].append(pd.Series(index=timestamps_for_dim, data=values_for_dim))
                line_dim += 1
                timestamps_for_dim = []
                values_for_dim = []
            elif line_state == TsTimestampLineState.EMPTY_DIM:
                if len(self.data) < line_dim + 1:
                    self.data.append([])
                self.data[line_dim].append(pd.Series(dtype="object"))
                line_dim += 1
            elif line_state == TsTimestampLineState.IN_CLASS_LABEL:
                if target is None:
                    target = ""
                target += token
            char_num += 1

        if line_state not in [TsTimestampLineState.IN_DIM, TsTimestampLineState.EMPTY_DIM,
                              TsTimestampLineState.IN_CLASS_LABEL]:
            raise TsFileParseException(
                f"Unexpected end of file in line {self.line_number}"
            )
        if not self.header[TsTag.CLASS_LABEL] and line_state == TsTimestampLineState.IN_CLASS_LABEL:
            raise TsFileParseException(
                f"File did not specify '@classLabel true' but a label was detected in line {self.line_number}"
            )
        if self.header[TsTag.CLASS_LABEL]:
            if line_state != TsTimestampLineState.IN_CLASS_LABEL:
                raise TsFileParseException(
                    f"File specified '@classLabel true' but no class label was provided in line {self.line_number}"
                )
            if target not in self.header[TsTag.CLASS_LABEL]:
                raise TsFileParseException(
                    f"The class value '{target}' on line {self.line_number} is not valid"
                )
            else:
                self.targets.append(target)

    def verify_metadata(self) -> None:
        """Verifies the parsed metadata of a `.ts` file by checking whether a full set of metadata has been provided.

        Returns:
            None
        """
        for k in self.required_meta_info:
            if self.header[k] is None:
                raise TsFileParseException(
                    f"Required header tag {k} is missing. A full set of metadata has not been provided before the data"
                )
        if self.header[TsTag.UNIVARIATE]:
            if self.header[TsTag.DIMENSIONS]:
                raise TsFileParseException(
                    "The '@dimensions' tag is not allowed if '@univariate true'"
                )
        else:
            if self.header[TsTag.DIMENSIONS] is None:
                raise TsFileParseException(
                    "The '@dimensions' tag is expected if '@univariate false'"
                )
        if self.header[TsTag.EQUAL_LENGTH]:
            if self.header[TsTag.SERIES_LENGTH] is None:
                raise TsFileParseException(
                    "The '@seriesLength' tag is required if '@equalLength true'"
                )
        else:
            if self.header[TsTag.SERIES_LENGTH]:
                raise TsFileParseException(
                    "The '@seriesLength' tag is not allowed if '@euqalLength false'"
                )

    def set_metadata(self):
        if self.header[TsTag.UNIVARIATE]:
            self.dim = 1
        else:
            self.dim = self.header[TsTag.DIMENSIONS]
        if self.header[TsTag.SERIES_LENGTH]:
            self.series_length = self.header[TsTag.SERIES_LENGTH]

    def parse(self):
        """Parses a `.ts` sktime formatted time series file.

        Returns:
            None
        """
        while (line := next(self.file, None)) is not None:
            line = line.strip()
            if self.state == self.State.PREFACE and line.startswith("@"):
                self.state = self.State.HEADER
            if self.state == self.State.HEADER:
                if line == "@data":
                    self.verify_metadata()
                    self.set_metadata()
                    if self.header[TsTag.TIMESTAMPS]:
                        self.state = self.State.BODY_TIME_STAMPS
                    else:
                        self.state = self.State.BODY
                else:
                    self.parse_header(line)
            elif self.state == self.State.BODY:
                self.parse_body(line)
            elif self.state == self.State.BODY_TIME_STAMPS:
                self.parse_body_timestamps(line)
            self.line_number += 1


class TsTimestampLineState(int, Enum):
    """
    Final state machine for handling the line parser state in `.ts` files that have timestamps.
    """
    START_LINE = auto()
    IN_DATA = auto()
    IN_DIM = auto()
    BEFORE_TUPLE = auto()
    START_TUPLE = auto()
    IN_TUPLE = auto()
    IN_CLASS_LABEL = auto()
    EMPTY_DIM = auto()

    def next(self, token: str) -> TsTimestampLineState:
        """Performs a state transition. A state transition can here consist of multiple atomic transitions in order.

        Args:
            token (str): The token at the current position.

        Returns (TsTimestampLineState):
            The new state after state transition.
        """
        token_restrictions: Dict[TsTimestampLineState, Optional[Tuple[bool, str]]] = {
            self.START_LINE: (True, "(:"),
            self.IN_DATA: None,
            self.IN_DIM: (True, ",:"),
            self.START_TUPLE: (False, "(),:"),
            self.IN_TUPLE: None,
            self.IN_CLASS_LABEL: (False, "(),:"),
            self.EMPTY_DIM: (True, "(:"),
            self.BEFORE_TUPLE: (True, "(")
        }
        if token_restrictions[self] is not None:
            white_list, tokens = token_restrictions[self]
            contains_token = token in tokens
            if (not contains_token and white_list) or (contains_token and not white_list):
                raise ValueError(
                    f"Unexpected token {token} state {self.name}"
                )
        if self == self.START_LINE:
            if token == "(":
                return self.START_TUPLE
            elif token == ":":
                return self.EMPTY_DIM
        if self == self.EMPTY_DIM and token == "(":
            return self.START_TUPLE
        if self == self.IN_TUPLE and token == ")":
            return self.IN_DIM
        if self == self.IN_DIM:
            if token == ":":
                return self.IN_DATA
            elif token == ",":
                return self.BEFORE_TUPLE
        if self == self.IN_DATA:
            if token == "(":
                return self.START_TUPLE
            elif token == ":":
                return self.EMPTY_DIM
            else:
                return self.IN_CLASS_LABEL
        if self == self.START_TUPLE:
            if token not in "(),:":
                return self.IN_TUPLE
        if self == self.BEFORE_TUPLE and token == "(":
            return self.START_TUPLE

        return self


def load_from_tsfile_to_dataframe(
        filepath,
        return_separate_x_and_y=True,
        return_class_labels=False,
        replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    filepath: str
        The full pathname of the .ts file to read.
    return_separate_x_and_y: bool
        true if X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data that contains class labels.
    return_class_labels: bool
        true if the class labels parsed from the header should be returned,
        false otherwise.
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    class_labels : ndarray
        The class labels. Only provided if return_class_labels is True.
    """
    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_int = None
    prev_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0

    # Parse the file
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "problemname tag requires an associated value"
                        )

                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True

                elif line.startswith("@timestamps"):

                    # Check that the data has not started

                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid

                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len != 2:
                        raise TsFileParseException(
                            "timestamps tag requires an associated Boolean " "value"
                        )

                    elif tokens[1] == "true":
                        timestamps = True

                    elif tokens[1] == "false":
                        timestamps = False

                    else:
                        raise TsFileParseException("invalid timestamps value")

                    has_timestamps_tag = True
                    metadata_started = True

                elif line.startswith("@univariate"):

                    # Check that the data has not started

                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid

                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len != 2:
                        raise TsFileParseException(
                            "univariate tag requires an associated Boolean  " "value"
                        )

                    elif tokens[1] == "true":
                        # univariate = True
                        pass

                    elif tokens[1] == "false":
                        # univariate = False
                        pass

                    else:
                        raise TsFileParseException("invalid univariate value")

                    has_univariate_tag = True
                    metadata_started = True

                elif line.startswith("@classlabel"):

                    # Check that the data has not started

                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid

                    tokens = line.split(" ")
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException(
                            "classlabel tag requires an associated Boolean  " "value"
                        )

                    if tokens[1] == "true":
                        class_labels = True

                    elif tokens[1] == "false":
                        class_labels = False

                    else:
                        raise TsFileParseException("invalid classLabel value")

                    # Check if we have any associated class values

                    if token_len == 2 and class_labels:
                        raise TsFileParseException(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )

                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True

                # Check if this line contains the start of data

                elif line.startswith("@data"):

                    if line != "@data":
                        raise TsFileParseException(
                            "data tag should not have an associated value"
                        )

                    if data_started and not metadata_started:
                        raise TsFileParseException("metadata must come before data")

                    else:
                        has_data_tag = True
                        data_started = True

                # If the 'data tag has been found then metadata has been
                # parsed and data can be loaded

                elif data_started:

                    # Check that a full set of metadata has been provided

                    if (
                            not has_problem_name_tag
                            or not has_timestamps_tag
                            or not has_univariate_tag
                            or not has_class_labels_tag
                            or not has_data_tag
                    ):
                        raise TsFileParseException(
                            "a full set of metadata has not been provided "
                            "before the data"
                        )

                    # Replace any missing values with the value specified

                    line = line.replace("?", replace_missing_vals_with)

                    # Check if we dealing with data that has timestamps

                    if timestamps:

                        # We're dealing with timestamps so cannot just split
                        # line on ':' as timestamps may contain one

                        has_another_value = False
                        has_another_dimension = False

                        timestamp_for_dim = []
                        values_for_dimension = []

                        this_line_num_dim = 0
                        line_len = len(line)
                        char_num = 0

                        while char_num < line_len:

                            # Move through any spaces

                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1

                            # See if there is any more data to read in or if
                            # we should validate that read thus far

                            if char_num < line_len:

                                # See if we have an empty dimension (i.e. no
                                # values)

                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dim + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dim].append(
                                        pd.Series(dtype="object")
                                    )
                                    this_line_num_dim += 1

                                    has_another_value = False
                                    has_another_dimension = True

                                    timestamp_for_dim = []
                                    values_for_dimension = []

                                    char_num += 1

                                else:

                                    # Check if we have reached a class label

                                    if line[char_num] != "(" and class_labels:

                                        class_val = line[char_num:].strip()

                                        if class_val not in class_label_list:
                                            raise TsFileParseException(
                                                "the class value '"
                                                + class_val
                                                + "' on line "
                                                + str(line_num + 1)
                                                + " is not "
                                                  "valid"
                                            )

                                        class_val_list.append(class_val)
                                        char_num = line_len

                                        has_another_value = False
                                        has_another_dimension = False

                                        timestamp_for_dim = []
                                        values_for_dimension = []

                                    else:

                                        # Read in the data contained within
                                        # the next tuple

                                        if line[char_num] != "(" and not class_labels:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                  "not "
                                                  "start "
                                                  "with a "
                                                  "'('"
                                            )

                                        char_num += 1
                                        tuple_data = ""

                                        while (
                                                char_num < line_len
                                                and line[char_num] != ")"
                                        ):
                                            tuple_data += line[char_num]
                                            char_num += 1

                                        if (
                                                char_num >= line_len
                                                or line[char_num] != ")"
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                  "not end"
                                                  " with a "
                                                  "')'"
                                            )

                                        # Read in any spaces immediately
                                        # after the current tuple

                                        char_num += 1

                                        while char_num < line_len and str.isspace(
                                                line[char_num]
                                        ):
                                            char_num += 1

                                        # Check if there is another value or
                                        # dimension to process after this tuple

                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False

                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False

                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True

                                        char_num += 1

                                        # Get the numeric value for the
                                        # tuple by reading from the end of
                                        # the tuple data backwards to the
                                        # last comma

                                        last_comma_index = tuple_data.rfind(",")

                                        if last_comma_index == -1:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has "
                                                  "no comma inside of it"
                                            )

                                        try:
                                            value = tuple_data[last_comma_index + 1:]
                                            value = float(value)

                                        except ValueError:
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does "
                                                  "not have a valid numeric "
                                                  "value"
                                            )

                                        # Check the type of timestamp that
                                        # we have

                                        timestamp = tuple_data[0:last_comma_index]

                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False

                                        except ValueError:
                                            timestamp_is_int = False

                                        if not timestamp_is_int:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True

                                            except ValueError:
                                                timestamp_is_timestamp = False

                                        # Make sure that the timestamps in
                                        # the file (not just this dimension
                                        # or case) are consistent

                                        if (
                                                not timestamp_is_timestamp
                                                and not timestamp_is_int
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                  "has an invalid timestamp '"
                                                + timestamp
                                                + "'"
                                            )

                                        if (
                                                previous_timestamp_was_int is not None
                                                and previous_timestamp_was_int
                                                and not timestamp_is_int
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                  "timestamp format is "
                                                  "inconsistent"
                                            )

                                        if (
                                                prev_timestamp_was_timestamp is not None
                                                and prev_timestamp_was_timestamp
                                                and not timestamp_is_timestamp
                                        ):
                                            raise TsFileParseException(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                  "timestamp format is "
                                                  "inconsistent"
                                            )

                                        # Store the values

                                        timestamp_for_dim += [timestamp]
                                        values_for_dimension += [value]

                                        #  If this was our first tuple then
                                        #  we store the type of timestamp we
                                        #  had

                                        if (
                                                prev_timestamp_was_timestamp is None
                                                and timestamp_is_timestamp
                                        ):
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if (
                                                previous_timestamp_was_int is None
                                                and timestamp_is_int
                                        ):
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True

                                        # See if we should add the data for
                                        # this dimension

                                        if not has_another_value:
                                            if len(instance_list) < (
                                                    this_line_num_dim + 1
                                            ):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(
                                                    timestamp_for_dim
                                                )

                                            instance_list[this_line_num_dim].append(
                                                pd.Series(
                                                    index=timestamp_for_dim,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dim += 1

                                            timestamp_for_dim = []
                                            values_for_dimension = []

                            elif has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                                                                "line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that "
                                      "is not followed by "
                                      "another tuple"
                                )

                            elif has_another_dimension and class_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                                                                "line "
                                    + str(line_num + 1)
                                    + " ends with a ':' while "
                                      "it should list a class "
                                      "value"
                                )

                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dim].append(
                                    pd.Series(dtype=np.float32)
                                )
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim

                            # If this is the 1st line of data we have seen
                            # then note the dimensions

                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim

                                if num_dimensions != this_line_num_dim:
                                    raise TsFileParseException(
                                        "line "
                                        + str(line_num + 1)
                                        + " does not have the "
                                          "same number of "
                                          "dimensions as the "
                                          "previous line of "
                                          "data"
                                    )

                        # Check that we are not expecting some more data,
                        # and if not, store that processed above

                        if has_another_value:
                            raise TsFileParseException(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                  "not followed by another "
                                  "tuple"
                            )

                        elif has_another_dimension and class_labels:
                            raise TsFileParseException(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it "
                                  "should list a class value"
                            )

                        elif has_another_dimension and not class_labels:
                            if len(instance_list) < (this_line_num_dim + 1):
                                instance_list.append([])

                            instance_list[this_line_num_dim].append(
                                pd.Series(dtype="object")
                            )
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim

                        # If this is the 1st line of data we have seen then
                        # note the dimensions

                        if (
                                not has_another_value
                                and num_dimensions != this_line_num_dim
                        ):
                            raise TsFileParseException(
                                "line " + str(line_num + 1) + " does not have the same "
                                                              "number of dimensions as the "
                                                              "previous line of data"
                            )

                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata

                        if class_labels and len(class_val_list) == 0:
                            raise TsFileParseException(
                                "the cases have no associated class values"
                            )

                    else:
                        dimensions = line.split(":")

                        # If first row then note the number of dimensions (
                        # that must be the same for all cases)

                        if is_first_case:
                            num_dimensions = len(dimensions)

                            if class_labels:
                                num_dimensions -= 1

                            for _dim in range(0, num_dimensions):
                                instance_list.append([])

                            is_first_case = False

                        # See how many dimensions that the case whose data
                        # in represented in this line has

                        this_line_num_dim = len(dimensions)

                        if class_labels:
                            this_line_num_dim -= 1

                        # All dimensions should be included for all series,
                        # even if they are empty

                        if this_line_num_dim != num_dimensions:
                            raise TsFileParseException(
                                "inconsistent number of dimensions. "
                                "Expecting "
                                + str(num_dimensions)
                                + " but have read "
                                + str(this_line_num_dim)
                            )

                        # Process the data for each dimension

                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))

                            else:
                                instance_list[dim].append(pd.Series(dtype="object"))

                        if class_labels:
                            class_val_list.append(dimensions[num_dimensions].strip())

            line_num += 1

    # Check that the file was not empty

    if line_num:
        # Check that the file contained both metadata and data

        if metadata_started and not (
                has_problem_name_tag
                and has_timestamps_tag
                and has_univariate_tag
                and has_class_labels_tag
                and has_data_tag
        ):
            raise TsFileParseException("metadata incomplete")

        elif metadata_started and not data_started:
            raise TsFileParseException("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise TsFileParseException("file contained metadata but no data")

        # Create a DataFrame from the data parsed above

        data = pd.DataFrame(dtype=np.float32)

        for dim in range(0, num_dimensions):
            data["dim_" + str(dim)] = instance_list[dim]

        # Check if we should return any associated class labels separately

        if class_labels:
            if return_separate_x_and_y:
                if return_class_labels:
                    return data, np.asarray(class_val_list), class_label_list
                else:
                    return data, np.asarray(class_val_list)

            else:
                data["class_vals"] = pd.Series(class_val_list)
                if return_class_labels:
                    return data, class_label_list
                else:
                    return data
        else:
            return data

    else:
        raise TsFileParseException("empty file")
