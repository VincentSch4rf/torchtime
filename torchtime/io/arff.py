from __future__ import annotations

import re

import numpy as np
import pandas as pd

from torchtime.exceptions import ArffFileParseException


def load_from_arff_to_dataframe(
        full_file_path_and_name,
        return_separate_labels=True,
        return_class_labels=False,
        replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    has_class_labels: bool
        true then line contains separated strings and class value contains
        list of separated strings, check for 'return_separate_X_and_y'
        false otherwise.
    return_separate_X_and_y: bool
        true then X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data.
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
    """
    instance_list = []
    class_val_list = []
    class_labels = None

    data_started = False
    is_multi_variate = False
    has_class_labels = False
    is_first_case = True

    # Parse the file
    with open(full_file_path_and_name, "r", encoding="utf-8") as f:
        for line in f:

            if line.strip():
                if (
                        is_multi_variate is False
                        and "@attribute" in line.lower()
                        and "relational" in line.lower()
                ):
                    is_multi_variate = True

                if (
                        has_class_labels is False
                        and "@attribute" in line.lower()
                        and ("target" in line.lower() or "classAttribute" in line.lower())
                ):
                    has_class_labels = True
                    pattern = re.compile("{([\w,]+)}")
                    result = pattern.search(line)
                    class_labels = result.group(1).split(',')
                    if len(set(class_labels)) != len(class_labels):
                        raise ArffFileParseException("Targets contain duplicate values!")

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information
                # has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        if has_class_labels:
                            line, class_val = line.split("',")
                            class_val_list.append(class_val.strip())
                        dimensions = line.split("\\n")
                        dimensions[0] = dimensions[0].replace("'", "")

                        if is_first_case:
                            for _d in range(len(dimensions)):
                                instance_list.append([])
                            is_first_case = False

                        for dim in range(len(dimensions)):
                            instance_list[dim].append(
                                pd.Series(
                                    [float(i) for i in dimensions[dim].split(",")]
                                )
                            )

                    else:
                        if is_first_case:
                            instance_list.append([])
                            is_first_case = False

                        line_parts = line.split(",")
                        if has_class_labels:
                            instance_list[0].append(
                                pd.Series(
                                    [
                                        float(i)
                                        for i in line_parts[: len(line_parts) - 1]
                                    ]
                                )
                            )
                            class_val_list.append(line_parts[-1].strip())
                        else:
                            instance_list[0].append(
                                pd.Series(
                                    [float(i) for i in line_parts[: len(line_parts)]]
                                )
                            )

    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(len(instance_list)):
        x_data["dim_" + str(dim)] = instance_list[dim]

    if return_separate_labels:
        return x_data, np.asarray(class_val_list), class_labels
    else:
        if has_class_labels:
            x_data["class_vals"] = pd.Series(class_val_list)
    return x_data, class_labels