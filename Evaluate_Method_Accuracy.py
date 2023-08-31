import os
import argparse
import pathlib
from os.path import join, exists
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tax_credit.plotting_functions import *
from tax_credit.eval_framework import *

def evaluate_method_accuracy(database_name, outdir):
    expected_results_dir = join("data/precomputed-results/", "mock-community")
    mock_results_fp = join(expected_results_dir, 'mock_results.tsv')
    results_dirs = [expected_results_dir]
    mock_dir = join("data", "mock-community")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    dataset_ids = ['mock-1', 'mock-2', 'mock-3', 'mock-4', 'mock-5', 'mock-6', 'mock-7', 'mock-8', 'mock-12', 'mock-13', 'mock-14', 'mock-15', 'mock-16', 'mock-18', 'mock-19', 'mock-20', 'mock-21', 'mock-22']
    method_ids = ['q2-NB', 'q2-SFM-RF', 'q2-SFM-SGD', 'q2-SFM-NB']
    ref_ids = [database_name]

    mock_results = evaluate_results(results_dirs, expected_results_dir, mock_results_fp, mock_dir,
                                    taxonomy_level_range=range(2, 7),
                                    min_count=1,
                                    taxa_to_keep=None,
                                    md_key='taxonomy',
                                    subsample=False,
                                    per_seq_precision=True,
                                    exclude=['other'],
                                    dataset_ids=dataset_ids,
                                    reference_ids=ref_ids,
                                    method_ids=method_ids,
                                    append=True,
                                    force=True,
                                    backup=False)

    color_palette = {
        'q2-NB': 'black',
        'q2-SFM-RF': 'darkgreen',
        'q2-SFM-SGD': 'red',
        'q2-SFM-NB': 'blue',
    }

    y_vars = ["Precision", "Recall", "F-measure", "Taxon Accuracy Rate", "Taxon Detection Rate"]
    pointplot_from_data_frame(mock_results, "Level", y_vars,
                              group_by="Reference", color_by="Method",
                              color_palette=color_palette, save_dir=outdir)

    metrics = ["Precision", "Recall", "F-measure", "Taxon Accuracy Rate", "Taxon Detection Rate"]
    heatmap_from_data_frame(mock_results, metrics, rows=["Method", "Parameters"], cols=["Reference", "Level"],
                            save_dir=outdir)

    mock_results_5 = mock_results[mock_results['Level'] == 5]
    for dataset in mock_results_5['Dataset'].unique():
        best = method_by_dataset_a1(mock_results_5, dataset)
        best.to_csv(join(outdir, '{0}-best_method.tsv'.format(dataset)), sep='\t')

    for method in mock_results_5['Method'].unique():
        top_params = parameter_comparisons(mock_results_5, method,
                                           metrics=['Taxon Accuracy Rate', 'Taxon Detection Rate', 'Precision', 'Recall', 'F-measure'])
        top_params[:5].to_csv(join(outdir, '{0}-top_params.tsv'.format(method)), sep='\t')

    level_ranges = [(4, 5), (5, 6), (6, 7)]
    for level_range in level_ranges:
        boxes = rank_optimized_method_performance_by_dataset(mock_results,
                                                             dataset="Reference",
                                                             metric="F-measure",
                                                             level_range=range(*level_range),
                                                             display_fields=["Method", "Parameters", "Taxon Accuracy Rate",
                                                                             "Taxon Detection Rate", "Precision", "Recall", "F-measure"],
                                                             paired=True,
                                                             parametric=True,
                                                             color=None,
                                                             color_palette=color_palette)
        for k, v in boxes.items():
            v.get_figure().savefig(join(outdir, 'level-{}-fmeasure-{}-boxplots.pdf'.format(level_range[0], k)),
                                   bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description='Evaluate method accuracy for different reference databases.')
    parser.add_argument('-r', '--reference_database_name', nargs='?', default='gg_13_8_otus',
                        help='Name of the reference database containing ref sequences and taxa.')
    parser.add_argument('-p', '--plots_path', nargs='?', type=pathlib.Path, default='plots',
                        help='Directory to save plots. Default: %(default)s')
    args = parser.parse_args()
    reference_database_name = args.reference_database_name
    plots_path = args.plots_path
    evaluate_method_accuracy(reference_database_name, plots_path)

if __name__ == '__main__':
    main()
