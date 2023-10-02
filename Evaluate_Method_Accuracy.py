import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tax_credit.plotting_functions import *
from tax_credit.eval_framework import *
 
def evaluate_method_accuracy(database_name, outdir):
    expected_results_dir = Path("data/precomputed-results/mock-community")
    mock_results_fp = expected_results_dir / 'mock_results.tsv'
    results_dirs = [expected_results_dir]
    mock_dir = Path("data/mock-community")
 
    outdir.mkdir(parents=True, exist_ok=True)
 
    #dataset_ids = [f"mock-{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]]
    dataset_ids = [f"mock-{i}" for i in [1, 2, 3]]
    method_ids = ['q2-NB', 'q2-TF']
    ref_ids = [database_name]
 
    mock_results = evaluate_results(
        results_dirs, expected_results_dir, mock_results_fp, mock_dir,
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
        backup=False
    )
 
    color_palette = {
        'q2-NB': 'black',
        'q2-TF': 'darkgreen',
    }
 
    y_vars = ["Precision", "Recall", "F-measure", "Taxon Accuracy Rate", "Taxon Detection Rate"]
    pointplot_from_data_frame(mock_results, "Level", y_vars,
                              group_by="Reference", color_by="Method",
                              color_palette=color_palette)
 
    metrics = y_vars
    heatmap_from_data_frame(mock_results, metrics, rows=["Method", "Parameters"], cols=["Reference", "Level"])
 
    mock_results_5 = mock_results[mock_results['Level'] == 5]
    for dataset in mock_results_5['Dataset'].unique():
        best = method_by_dataset_a1(mock_results_5, dataset)
        best.to_csv(outdir / f'{dataset}-best_method.tsv', sep='\t')
 
    for method in mock_results_5['Method'].unique():
        top_params = parameter_comparisons(mock_results_5, method, metrics=y_vars)
        top_params[:5].to_csv(outdir / f'{method}-top_params.tsv', sep='\t')
 
    level_ranges = [(4, 5), (5, 6), (6, 7)]
    for level_range in level_ranges:
        boxes = rank_optimized_method_performance_by_dataset(
            mock_results, dataset="Reference", metric="F-measure",
            level_range=range(*level_range),
           display_fields=y_vars + ["Method", "Parameters"],
            paired=True, parametric=True, color=None, color_palette=color_palette
        )
        for k, v in boxes.items():
            v.get_figure().savefig(outdir / f'level-{level_range[0]}-fmeasure-{k}-boxplots.pdf', bbox_inches='tight')
            
def main():
    parser = argparse.ArgumentParser(description='Evaluate method accuracy for different reference databases.')
    parser.add_argument('-r', '--reference_database_name', default='gg_13_8_otus',
                        help='Name of the reference database containing ref sequences and taxa.')
    parser.add_argument('-p', '--plots_path', type=Path, default='plots',
                        help='Directory to save plots. Default: %(default)s')
    args = parser.parse_args()
   
    evaluate_method_accuracy(args.reference_database_name, args.plots_path)
    
if __name__ == '__main__':
    main()
