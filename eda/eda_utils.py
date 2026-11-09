"""Общие пути, стиль графиков и недельная таблица таргета для ноутбуков EDA."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

EDA_DIR = Path(__file__).resolve().parent
DATA = EDA_DIR / "data" / "processed"

# Категориальные цвета - в фиксированном порядке, не по рангу серии
SERIES = [
    "#2a78d6",  # синий
    "#eb6834",  # оранжевый
    "#1baf7a",  # аква
    "#eda100",  # жёлтый
    "#e87ba4",  # маджента
    "#008300",  # зелёный
    "#6250d6",  # фиолетовый
    "#e34948",  # красный
]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#b5b4ae"
# Расходящаяся шкала: синий - серый ноль - красный (для корреляций)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#104281", "#3987e5", "#f0efec", "#e66767", "#a02a2a"],
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq",
    ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)


def setup_style() -> None:
    """Тонкие линии, сплошная бледная сетка, нейтральные подписи."""

    mpl.rcParams.update(
        {
            "figure.figsize": (11, 4),
            "figure.dpi": 110,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": NEUTRAL,
            "axes.labelcolor": INK_MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "text.color": INK,
            "savefig.bbox": "tight",
        },
    )
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 160)


def load(relative: str) -> pd.DataFrame:
    """Прочитать parquet из `eda/data/processed`, например `eia/eia_weekly_features.parquet`."""

    return pd.read_parquet(DATA / relative)


def report_week(release_date: pd.Series) -> pd.Series:
    """
    Неделя отчёта EIA (пятница) по дате релиза.

    Отчёт выходит в среду за неделю, закончившуюся в пятницу пятью днями раньше;
    праздники сдвигают релиз на четверг, пятницу или следующий понедельник.
    Правило «последняя пятница не позже релиза минус 5 дней» верно для всех
    этих сдвигов (понедельник 2025-12-29 - отчёт за 2025-12-19).
    """

    shifted = release_date - pd.Timedelta(days=5)
    return shifted - pd.to_timedelta((shifted.dt.day_of_week - 4) % 7, unit="D")


def release_frame() -> pd.DataFrame:
    """
    Недельная таблица таргета: релиз EIA, консенсус, факт, surprise и API.

    Колонки:
        - release_date: дата публикации EIA (Myfxbook);
        - period: пятница - конец недели отчёта;
        - actual_mmbbl: изменение коммерческих запасов в первой публикации, млн барр.;
        - consensus_mmbbl: консенсус Myfxbook, млн барр.;
        - surprise_mmbbl: actual - consensus (>0 - запасы выросли сильнее ожиданий);
        - eia_change_mmbbl: то же изменение по текущим (пересмотренным) данным EIA API;
        - api_mmbbl: изменение запасов по API (Trendonify), выходит во вторник до EIA.
    """

    consensus = load("consensus/myfxbook_eia_consensus.parquet")
    eia = load("eia/eia_weekly_features.parquet")
    api = load("consensus/trendonify_api_crude_full_history.parquet")
    frame = consensus.rename(columns={"inventory_surprise_mmbbl": "surprise_mmbbl"})[
        ["release_date", "actual_mmbbl", "consensus_mmbbl", "surprise_mmbbl"]
    ].assign(period=lambda df: report_week(df["release_date"]))
    frame = frame.merge(
        eia[["period", "crude_stocks_change_mmbbl"]].rename(
            columns={"crude_stocks_change_mmbbl": "eia_change_mmbbl"},
        ),
        on="period",
        how="left",
    ).merge(
        api[["reference_week_end", "api_actual_million"]].rename(
            columns={"reference_week_end": "period", "api_actual_million": "api_mmbbl"},
        ),
        on="period",
        how="left",
    )
    return frame.sort_values("release_date").reset_index(drop=True)
