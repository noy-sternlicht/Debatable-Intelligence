import argparse
import json
import os
from scipy.stats import pearsonr

import pandas as pd

from util import setup_default_logger, read_annotation_data, agg_model_predictions, compute_kendall_tau


def avg_all_annotators(annotations: pd.DataFrame, annotations_col_name) -> float:
    all_annotations = []
    nr_annotators = 0
    for index, row in annotations.iterrows():
        all_annotations.extend(row[annotations_col_name])
        nr_annotators += len(row[annotations_col_name])

    annotations_sum = sum(all_annotations)

    return annotations_sum / nr_annotators


def main():
    data = read_annotation_data(args.data_path)
    unique_sources = data['source'].unique()

    results = []
    eval_models_config = json.load(open(args.eval_models_config))['eval_models']
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

        judge_averages = {'Judge': model_info['name']}
        for model_agg_func, agg_model_pred in model_pred.items():
            for source in unique_sources:
                source_data = data[data['source'] == source]
                source_predictions = [agg_model_pred[row_id] for row_id in source_data['id']]
                source_average = sum(source_predictions) / len(source_predictions)
                judge_averages[source] = source_average
            results.append(judge_averages)

    human_averages = {}
    for source in unique_sources:
        logger.info(f'Processing source: {source}')
        source_annotations = data[data['source'] == source]
        source_average = avg_all_annotators(source_annotations, args.col_name)
        human_averages[source] = source_average

    sources_sorted = sorted(human_averages.keys(), key=lambda x: human_averages[x])
    human_sources_values = [human_averages[source] for source in sources_sorted]
    human_averages['Judge'] = 'Human'
    results.append(human_averages)
    avg_results_per_source_df = pd.DataFrame(results)

    avg_results_per_source_df = avg_results_per_source_df[['Judge'] + sources_sorted]
    avg_results_per_source_df['tau-c'] = avg_results_per_source_df.apply(
        lambda row: compute_kendall_tau(row[sources_sorted].values.tolist(), human_sources_values, variant='c'), axis=1
    )



    avg_results_per_source_df['pearson'] = avg_results_per_source_df.apply(
        lambda row: pearsonr(row[sources_sorted].values.tolist(), human_sources_values)[0], axis=1
    )

    # for _, row in avg_results_per_source_df.iterrows():
    #     print(row[sources_sorted].values.tolist())
    #     print(human_sources_values)


    avg_results_per_source_df = avg_results_per_source_df.round(3)
    avg_results_per_source_path = os.path.join(args.output_path, 'avg_results_per_source.csv')
    avg_results_per_source_df.to_csv(avg_results_per_source_path, index=False)
    logger.info(f'Wrote avg results per source to {avg_results_per_source_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_models_config', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--col_name', default='goodopeningspeech', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)

    main()
