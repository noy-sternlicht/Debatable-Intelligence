import argparse
import json
import os
import pandas as pd

from util import setup_default_logger, read_annotation_data, agg_model_predictions, get_model_size, get_model_family


def main():
    logger.info(f'loading eval models config from {args.eval_models_config}')
    eval_models_config = json.load(open(args.eval_models_config))['eval_models']

    scores_histograms = []
    parsability_histograms = []

    for model_info in eval_models_config:
        logger.info(f'loading results for model {model_info["name"]}')
        model_pred = {}
        prompt = args.prompt
        for model_name, res_dir in model_info['results'].items():
            res_path = os.path.join(res_dir, f'{prompt}.csv')
            if not os.path.exists(res_path):
                logger.warning(f'Results file {res_path} does not exist. Skipping.')
                break
            res = read_annotation_data(res_path)
            pred_col_postfix = f'{model_name}_goodopeningspeech_{prompt}'
            pred_col = [c for c in res.columns if c.endswith(pred_col_postfix)]
            if len(pred_col) == 0:
                logger.warning(f'No prediction column found for model {model_name} in {res_path}. Skipping.')
                break
            for _, row in res.iterrows():
                if row['id'] not in model_pred:
                    model_pred[row['id']] = []
                model_pred[row['id']].append(row[pred_col[0]])

        if len(model_pred) == 0:
            logger.warning(f'No predictions found for model {model_info["name"]}. Skipping.')
            continue

        if len(model_info['results']) > 1:
            model_pred = agg_model_predictions(model_pred)
        else:
            model_pred = {'-': {row_id: row[0] for row_id, row in model_pred.items()}}

        for model_agg_func, agg_model_pred in model_pred.items():
            model_histogram = {'model_name': model_info['name'],
                               'model_family': get_model_family(model_info['name']),
                               'model_size': get_model_size(model_info['name']),
                               'model_agg_func': model_agg_func,
                               '-1': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            model_pars_histogram = {'model_name': model_info['name'],
                                    'model_family': get_model_family(model_info['name']),
                                    'model_size': get_model_size(model_info['name']),
                                    'model_agg_func': model_agg_func,
                                    'parsable': 0, 'non_parsable': 0}
            for _, score in agg_model_pred.items():
                score = str(int(score))
                if score not in model_histogram:
                    logger.warning(f'Unknown score {score} for model {model_info["name"]}')
                    model_histogram['-1'] += 1
                    model_pars_histogram['non_parsable'] += 1
                    continue

                model_histogram[score] += 1
                if int(score) == -1:
                    model_pars_histogram['non_parsable'] += 1
                else:
                    model_pars_histogram['parsable'] += 1

            scores_histograms.append(model_histogram)
            parsability_histograms.append(model_pars_histogram)

    all_human_agg_scores = []
    all_human_scores = []
    all_annotations = read_annotation_data('data_splits/all.csv')
    for _, row in all_annotations.iterrows():
        row_annotations = row['goodopeningspeech']
        rounded_avg = round(sum(row_annotations) / len(row_annotations))
        all_human_agg_scores.append(rounded_avg)
        for score in row_annotations:
            all_human_scores.append(score)

    all_human_avg = sum(all_human_scores) / len(all_human_scores)
    logger.info(f'All human average score: {all_human_avg}')

    human_scores_hist = {'model_name': 'Human',
                         'model_family': 'Human',
                         'model_size': 1,
                         'model_agg_func': 'mean',
                         '-1': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    for score in all_human_scores:
        score = str(int(score))
        if score not in human_scores_hist:
            logger.warning(f'Unknown score {score} for human')
            continue

        human_scores_hist[score] += 1
    scores_histograms.append(human_scores_hist)

    scores_histograms_df = pd.DataFrame(scores_histograms)
    scores_histograms_df = scores_histograms_df.set_index(
        ['model_family', 'model_size', 'model_name', 'model_agg_func'])
    scores_histograms_df = scores_histograms_df.div(scores_histograms_df.sum(axis=1), axis=0) * 100
    scores_histograms_df = scores_histograms_df.round(2)
    scores_histograms_df = scores_histograms_df.sort_index()

    results_path = os.path.join(args.output_path, 'scores_hist.csv')
    scores_histograms_df.to_csv(results_path)
    logger.info(f'Saved scores histograms to {results_path}')

    parsability_histograms_df = pd.DataFrame(parsability_histograms)
    parsability_histograms_df = parsability_histograms_df.set_index(
        ['model_family', 'model_size', 'model_name', 'model_agg_func'])
    parsability_histograms_df = parsability_histograms_df.sort_index()
    parsability_results_path = os.path.join(args.output_path, 'parsability_hist.csv')
    parsability_histograms_df.to_csv(parsability_results_path)
    logger.info(f'Saved parsability histograms to {parsability_results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_models_config', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)

    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)

    args = parser.parse_args()

    main()
