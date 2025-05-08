import argparse
import json
from typing import Dict

from util import setup_default_logger, read_annotation_data, aggregate_human_annotations_avg, compute_kendall_tau, \
    agg_model_predictions, get_model_size, get_model_family
import pandas as pd
import os


def evaluate_model_predictions(predictions: pd.Series, human_annotations: pd.Series) -> Dict[str, float]:
    kendall_tau_c = compute_kendall_tau(predictions, human_annotations, variant='c')
    eval_results = {'Tau-c': kendall_tau_c}

    return eval_results


def avg_all_annotators(annotations: pd.DataFrame, annotations_col_name) -> float:
    all_annotations = []
    nr_annotators = 0
    for index, row in annotations.iterrows():
        all_annotations.extend(row[annotations_col_name])
        nr_annotators += len(row[annotations_col_name])

    annotations_sum = sum(all_annotations)

    return annotations_sum / nr_annotators


def evaluate_predicted_scores():
    logger.info(f'loading eval models config from {args.eval_models_config}')
    eval_models_config = json.load(open(args.eval_models_config))['eval_models']
    all_results = []
    human_annotations = read_annotation_data(args.data_path)

    for model_info in eval_models_config:
        logger.info(f'Evaluating model {model_info["name"]}')
        model_pred = {}
        prompt = args.prompt
        for model_name, res_dir in model_info['results'].items():
            results_dir_path = os.path.join(res_dir, f'{prompt}.csv')
            if not os.path.exists(results_dir_path):
                logger.warning(f'Results directory {results_dir_path} does not exist. Skipping model {model_name}.')
                break
            res = read_annotation_data(results_dir_path)
            pred_col_postfix = f'{model_name}_{args.col_name}_{prompt}'
            pred_col = [c for c in res.columns if c.endswith(pred_col_postfix)]
            if len(pred_col) == 0:
                logger.warning(
                    f'No predictions found for model {model_name} with postfix {pred_col_postfix}. Skipping.')
                continue
            for _, row in res.iterrows():
                if row['id'] not in model_pred:
                    model_pred[row['id']] = []
                model_pred[row['id']].append(row[pred_col[0]])

        if not model_pred:
            logger.warning(f'No predictions found for model {model_info["name"]}. Skipping.')
            continue

        if len(model_info['results']) > 1:
            model_pred = agg_model_predictions(model_pred)
        else:
            model_pred = {'-': {row_id: row[0] for row_id, row in model_pred.items()}}

        for model_agg_func, agg_model_pred in model_pred.items():
            col_name = args.col_name
            gold_col = f'{col_name}_avg'
            human_annotations[gold_col] = human_annotations[col_name].apply(aggregate_human_annotations_avg)
            model_pred_ordered = [agg_model_pred[row_id] for row_id in human_annotations['id']]
            results_agg = evaluate_model_predictions(pd.Series(model_pred_ordered), human_annotations[gold_col])

            all_results.append({'model_family': get_model_family(model_info['name']),
                                'model_size': get_model_size(model_info['name']),
                                'model_agg_func': model_agg_func,
                                'model': model_info['name'], 'prompt': prompt,
                                **results_agg})

    all_results_df = pd.DataFrame(all_results)
    all_results_df = all_results_df.round(3)
    all_results_df = all_results_df.sort_values(by=['model_family', 'model_size', 'model_agg_func'])
    all_results_path = os.path.join(args.output_path, 'all_results.csv')
    all_results_df.to_csv(all_results_path, index=False)
    logger.info(f'Wrote all results to {all_results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_models_config', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--col_name', default='goodopeningspeech', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)

    evaluate_predicted_scores()
