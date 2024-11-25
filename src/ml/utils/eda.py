"""Modules for Exploratory data analysis plotting and printing."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def describe_general_data(
    data: pd.DataFrame,
    missing_value: int = 5,
) -> None:
    """Prints some generic information about the dataframe
    Shape of Dataframe
    Sum of Duplicated Rows (for all attributes)
    Sum of unique dtypes
    Top X Missing Values sorted by Sum,

    Args:
        data (pd.Dataframe): entire dataframe
        missing_value (int, optional): Top X missing values. Defaults to 5.

    ** start of sample output**
    General Data Information
    ------------------------------
    Shape of Dataframe:           (199463, 42)
    Sum of Duplicated Rows:       0
    ------------------------------
    Summary of Columns dtypes     42 total cols below
    object    29
    int64     13
    Name: count, dtype: int64
    ------------------------------
    Top 5 Missing Values            874 overall na values
    hispanic_origin                  874
    id                                 0
    tax_filer_stat                     0
    family_members_under_18            0
    own_business_or_self_employed      0
    dtype: int64
    ** end of sample output**


    """
    dtypes_value = data.dtypes.value_counts()
    na_value = data.isna().sum()

    print("General Data Information", "-" * 30, sep="\n")
    print(f"{'Shape of Dataframe:':30}{data.shape}")
    print(f"{'Sum of Duplicated Rows:':30}{data.duplicated().sum()}")
    print("-" * 30)
    print(f"{'Summary of Columns dtypes':30}{dtypes_value.sum()} total cols below")
    print(dtypes_value)
    print("-" * 30)
    print(f"{f'Top {missing_value} Missing Rows':30}{na_value.sum()} overall na values")
    print(na_value.sort_values(ascending=False).head(missing_value))


def show_value_count_and_percentage(
    data: pd.DataFrame,
    label: str,
    drop_na: bool = False,
    top_5=5,
    as_df=False,
):
    # Reference from Example 3, https://www.statology.org/pandas-value_counts-percentage/
    """Produce a dataframe of value counts with count and percentage

    Args:
        data (pd.Dataframe): Dataframe of the data
        label (str): attribute name in string
        drop_na (bool, optional): Whether to drop na before value count?. Defaults to False.
        top_5 (int, optional): df.head(top_5). Defaults to 5.
        as_df (bool, optional): True returns dataframe. Defaults to False.

    Returns:
        Print output if False, dataframe if True
    """

    counts = data[label].value_counts(dropna=drop_na)
    percent = (
        data[label].value_counts(normalize=True, dropna=drop_na).multiply(100).round(2)
    )
    result = pd.concat([counts, percent], axis=1, keys=["count", "percentage %"])
    result = result.head(top_5)
    if as_df:
        return result
    print(result)


def plot_cat_distribution(
    data: pd.DataFrame,
    label: str,
    title: str = None,
    figsize: tuple[int, int] = (5, 4),
    dropna: bool = False,
    rot: int = 0,
) -> mpl.axes._axes.Axes:
    """Plot categorical bar chart using Pandas dataframe and Matplotlib.

    Args:
        data (pd.DataFrame): DataFrame of the data
        label (str): a string of the label you wish to plot
        title (str, optional): Defaults to "Count of {label}. Defaults to None.
        figsize (tuple[int,int], optional): Defaults to (5, 4).
        dropna (bool, optional): Whether to plot or hide NA. Defaults to False.
        rot (int, optional): rotation of xlabel. Defaults to 0.

    Returns:
        mpl.axes._axes.Axes: _description_
    """

    plot_data = data[label].value_counts(dropna=dropna).sort_values(ascending=True)
    column_list = data[label].unique()

    # extends figure height if there are too many category
    if len(column_list) > 12:
        height = len(column_list) * 0.18
        figsize = (5, height)

    if len(column_list) > 60:
        print(
            f"This label has {len(column_list)}, too much categories to plot a bar chart.",
            end="\n\n",
        )
        return

    # check if len of the str is too long? then use horizontal bar instead of vertical
    # added lambda to deal with nan type float. convert it to str of len3
    if len(max(column_list, key=lambda x: len(str(x)))) > 10:
        axe = plot_data.plot.barh(rot=rot, figsize=figsize)
    else:
        axe = plot_data.plot.bar(rot=rot, figsize=figsize)

    if title:
        plt.title(title)
    else:
        plt.title(f"count of {label}")

    return axe


def plot_single_numeric_hist(
    data: pd.DataFrame,
    label: str,
    bins: int = 30,
    figsize: tuple[int] = (4, 3),
    **kwargs,
) -> mpl.axes._axes.Axes:
    """
    Plot histogram using pandas dataframe and matplotlib library.
    Test different bin size for different histogram results

    Args:
        data (pd.DataFrame): dataframe of the data
        label (str): attribute in string that you wish to plot
        bins (int, optional): Bins to appear in dataframe. Defaults to 30.
        figsize (tuple[int], optional): Defaults to (4, 3).

    Returns:
        mpl.axes._axes.Axes: return as an axes for customization if needed
    """
    axe = data[label].hist(bins=bins, edgecolor="k", figsize=figsize, **kwargs)
    axe.set_title(f"Distribution of {label}")
    axe.set_xlabel(label)
    axe.set_ylabel(f"count of {label}")
    plt.grid(axis="x")

    return axe


def eda_for_single_numeric(
    data: pd.DataFrame,
    label: str,
    edit_plot: bool = False,
    top_5: int = 5,
) -> mpl.axes._axes.Axes:
    """A wrapper for a relevant EDA information, related to numeric attribute,
    either use it on single attribute or wrap it in a manual loop for all numeric attribute

    Args:
        data (pd.DataFrame): DataFrame of the data
        label (str): Single Attribute as a String
        edit_plot (bool, optional): Returns mpl.axes for customization. Defaults to False.
        top_5 (int, optional): df.head(top_5). Defaults to 5.

    Returns:
        mpl.axes._axes.Axes: axes for further customization of plt is needed.
    """
    print(f"Information for {label}".upper())
    print("-" * 30)
    print("Describe Statistic")
    print(data[label].describe().round(1))
    print("-" * 30)
    print("Showing sum of nulls: ", data[label].isna().sum())
    print("-" * 30)
    print("Showing top 5 frequency value count")
    show_value_count_and_percentage(data, label, as_df=False, top_5=top_5)
    print("-" * 30)

    fig, axe = plt.subplots(1, 2, width_ratios=(2, 1))
    plot_single_numeric_hist(data, label, figsize=(7, 3), ax=axe[0])
    data.boxplot(column=label, ax=axe[1], ylabel=f"{label} range for boxplot")
    fig.subplots_adjust(wspace=0.3)
    if edit_plot:
        return axe
    plt.tight_layout()
    plt.show()


def eda_for_single_category_str(
    data: pd.DataFrame,
    label: str,
    edit_plot: bool = False,
    top_5: int = 5,
) -> mpl.axes._axes.Axes:
    """
    A wrapper for a relevant EDA information, related to string/object dtype attribute,
    either use it on single attribute or wrap it in a manual loop
    for all string/object dtype attribute

    Args:
        data (pd.DataFrame): DataFrame of the data
        label (str): Single Attribute as a String
        edit_plot (bool, optional): Returns mpl.axes for customization. Defaults to False.
        top_5 (int, optional): df.head(top_5). Defaults to 5.

    Returns:
        mpl.axes._axes.Axes: axes for further customization of plt is needed.
    """
    print(f"Information for {label}")
    print("-" * 30)
    print(f"Number of nunique values: {data[label].nunique()}")
    print("-" * 30)
    print("Showing sum of nulls: ", data[label].isna().sum())
    print("-" * 30)
    print("Showing top 5 frequency value count")
    show_value_count_and_percentage(data, label, as_df=False, top_5=top_5)
    print("-" * 30)
    axe = plot_cat_distribution(data, label)

    if edit_plot:
        return axe
    plt.show()


def eda_all_numeric_attribute(data):
    """Wrapper to loop through all numeric attribute"""
    for col in data.select_dtypes(include=np.number):
        eda_for_single_numeric(data=data, label=col)


def eda_all_category_attribute(data):
    """Wrapper to loop through all category attribute"""
    for col in data.select_dtypes(include="object"):
        eda_for_single_category_str(data=data, label=col)
