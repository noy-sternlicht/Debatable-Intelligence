import ast
import logging
import os
import string
from datetime import datetime
import nltk
from typing import List, Dict

import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr, kendalltau, zscore
import krippendorff
from colorama import Fore, Style

import pandas as pd


def get_model_family(model_name):
    model_family = 'human'
    if '+' in model_name.lower():
        model_family = 'z-ensemble'
    elif 'gpt-4' in model_name.lower():
        model_family = 'GPT'
    elif 'o4' in model_name.lower() or 'o3' in model_name.lower():
        model_family = 'o'
    elif 'claude' in model_name.lower():
        model_family = 'Claude'
    elif 'llama' in model_name.lower():
        model_family = 'Llama'
    elif 'qwen' in model_name.lower():
        model_family = 'Qwen'

    return model_family


def get_model_size(model_name):
    if model_name == 'GPT-4.1-nano':
        return 1.0
    if model_name == 'GPT-4o-mini':
        return 2.0
    if model_name == 'GPT-4.1-mini':
        return 3.0
    if model_name == 'GPT-4o':
        return 4.0
    if model_name == 'GPT-4.1':
        return 5.0
    if model_name == 'o3-mini':
        return 1.0
    if model_name == 'o4-mini':
        return 2.0
    if model_name == 'o3':
        return 3.0
    if 'haiku' in model_name.lower():
        return 1.0
    if '3.5_sonnet' in model_name.lower():
        return 2.0
    if '3.7_sonnet' in model_name.lower():
        return 3.0
    if model_name == 'human':
        return 1.0
    if '+' in model_name:
        return 1.0

    model_size = model_name.split('B')[0].split('-')[-1]
    return float(model_size)


def agg_model_predictions(model_pred: Dict[str, list]) -> Dict[str, Dict[str, int]]:
    agg_predictions = {'rounded_avg': {}}
    for key, values in model_pred.items():
        for agg_func in agg_predictions.keys():
            if key not in agg_predictions[agg_func]:
                agg_predictions[agg_func][key] = -1
            agg_val = round(sum(values) / len(values))
            agg_predictions['rounded_avg'][key] = agg_val
    return agg_predictions


def aggregate_human_annotations_majority(annotations: pd.Series) -> int:
    majority_vote = max(set(annotations), key=annotations.count)
    return majority_vote


def aggregate_human_annotations_rounded_avg(annotations: pd.Series) -> int:
    return round(sum(annotations) / len(annotations))


def aggregate_human_annotations_avg(annotations: pd.Series, normalize=False) -> float:
    if normalize:
        annotations = zscore(annotations)

    avg = sum(annotations) / len(annotations)
    return avg


def read_annotation_data(annotations_path: str) -> pd.DataFrame:
    annotations = pd.read_csv(annotations_path)
    list_cols = ['labeler_ids', 'goodopeningspeech']
    for col in list_cols:
        nr_nans = annotations[col].isna().sum()
        if not nr_nans:
            annotations[col] = annotations[col].apply(ast.literal_eval)

    # annotations = annotations.dropna()

    return annotations


def create_out_dir(out_dir: str) -> None:
    """Create a directory if it does not exist"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


class ColoredFormatter(logging.Formatter):
    """Custom formatter to color log levels in the console."""

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL
        record.levelname = f"{log_color}{record.levelname}{reset}"  # Color the level name
        return super().format(record)


def setup_default_logger(output_dir: str) -> logging.Logger:
    """Set up a logger that writes to output_dir with colored console output"""
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

    log_format = "[%(levelname)s] [%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

    # Create formatters
    file_formatter = logging.Formatter(log_format)  # Standard formatter for files
    colored_formatter = ColoredFormatter(log_format)  # Colored formatter for console

    # Console Handler (prints to terminal with colors)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(colored_formatter)
    stream_handler.setLevel(logging.DEBUG)

    # File Handler (writes logs to a file)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(logs_dir, f'{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Attach handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f'Writing logs to {log_file}')
    return logger


def get_unique_values(annotations1, annotations2):
    unique_values_1 = set(annotations1)
    unique_values_2 = set(annotations2)
    return unique_values_1, unique_values_2


def compute_pearson(annotations1, annotations2, undefined_value=None, normalize=False):
    unique_values_1, unique_values_2 = get_unique_values(annotations1, annotations2)
    if len(unique_values_1) == 1 or len(unique_values_2) == 1:
        return undefined_value

    if normalize:
        annotations1 = zscore(annotations1)
        annotations2 = zscore(annotations2)
    pearson_correlation, _ = pearsonr(annotations1, annotations2)
    return pearson_correlation


def compute_spearman(annotations1, annotations2, undefined_value=None, normalize=False):
    unique_values_1, unique_values_2 = get_unique_values(annotations1, annotations2)
    if len(unique_values_1) == 1 or len(unique_values_2) == 1:
        return undefined_value

    if normalize:
        annotations1 = zscore(annotations1)
        annotations2 = zscore(annotations2)

    spearman_correlation, _ = spearmanr(annotations1, annotations2)
    return spearman_correlation


def compute_kendall_tau(annotations1, annotations2, variant, undefined_value=None, normalize=False):
    unique_values_1, unique_values_2 = get_unique_values(annotations1, annotations2)
    if len(unique_values_1) == 1 or len(unique_values_2) == 1:
        return undefined_value

    if normalize:
        annotations1 = zscore(annotations1)
        annotations2 = zscore(annotations2)

    tau, _ = kendalltau(annotations1, annotations2, variant=variant)
    return tau


def compute_kappa_agreement(annotations1, annotations2, undefined_value=None):
    unique_values_1, unique_values_2 = get_unique_values(annotations1, annotations2)
    if len(unique_values_1) == 1 or len(unique_values_2) == 1:
        return undefined_value
    labels = [-1, 1, 2, 3, 4, 5]
    for val in unique_values_1.union(unique_values_2):
        if val not in labels:
            return undefined_value
    for val in unique_values_2:
        if val not in labels:
            return undefined_value
    kappa = cohen_kappa_score(annotations1, annotations2, weights='quadratic', labels=[-1, 1, 2, 3, 4, 5])
    return kappa


def compute_krippendorff_alpha(annotations1, annotations2, undefined_value=None, normalize=False):
    unique_values_1, unique_values_2 = get_unique_values(annotations1, annotations2)
    if len(unique_values_1) == 1 or len(unique_values_2) == 1:
        return undefined_value

    if normalize:
        annotations1 = zscore(annotations1)
        annotations2 = zscore(annotations2)

    reliability_data_matrix = np.array([annotations1.values, annotations2.values])
    return krippendorff.alpha(reliability_data_matrix, level_of_measurement='interval')


def compute_mae(annotations1: pd.Series, annotations2: pd.Series, undefined_value=None, normalize=False):
    if normalize:
        unique_values_1, unique_values_2 = get_unique_values(annotations1, annotations2)
        if len(unique_values_1) == 1 or len(unique_values_2) == 1:
            return undefined_value
        annotations1 = zscore(annotations1)
        annotations2 = zscore(annotations2)
    mae = (annotations1 - annotations2).abs().mean()
    return mae


def word_tokenize_text(text: str) -> List[str]:
    word_tokens = nltk.word_tokenize(text)
    # Undoing the tokenization of opening and closing quotes
    word_tokens = [word.replace("``", '"').replace("''", '"') for word in word_tokens]
    word_tokens = [word for word in word_tokens if word not in string.punctuation]
    return word_tokens


def sent_tokenize_text(text: str) -> List[str]:
    paragraphs = text.split('\n')
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(nltk.sent_tokenize(paragraph))
    return sentences
