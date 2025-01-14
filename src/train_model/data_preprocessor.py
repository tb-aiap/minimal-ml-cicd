"""Module for data cleaning and feature engineering."""

# import logging

# import omegaconf
# import pandas as pd

# from . import data_model

# logger = logging.getLogger(__name__)
# COL = data_model.ColumnEnum


# class DataPreprocessor:
#     ...
#     """For feature engineering and additional preprocessing."""

#     def __init__(self, params: omegaconf.DictConfig):

#         self.params = params

#     def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
#         # one hot encode columns
#         return data

#     def feature_engineer(self, data: pd.DataFrame) -> pd.DataFrame:

#         data = self._fe_bin_storey_range_above_25(data)

#         return data

#     def _fe_bin_storey_range_above_25(self, data: pd.DataFrame):
#         """combine some categories into a similar group
#         sample output: 13 TO 15
#         for anything above X, is group into Above X
#         for anything between 19 TO 21 and 22 TO 24 is grouped into 19 TO 24
#         """

#         # each row input is 04 TO 06
#         def _bin_storey(row):
#             group_above = 25
#             additional_group = [19, 22]
#             start_storey = int(row.split()[0])

#             # additional feature engineering after multivariate exploration
#             if start_storey in additional_group:
#                 return "19 TO 24".format(group_above)

#             if start_storey >= group_above:
#                 return "Above {}".format(group_above)
#             return row

#         data["storey_range"] = data[COL.storey_range].apply(_bin_storey)

#         return data
