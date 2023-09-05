import os
import argparse
import pathlib
from os.path import join, exists, split, sep
from os import makedirs, getpid
from glob import glob
from shutil import rmtree
from pandas import DataFrame
import csv
import json
import tempfile
from itertools import product
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from qiime2.plugins import feature_classifier
from qiime2 import Artifact
from qiime2.plugin import Int, Str, Float, Bool, Choices, Range
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from q2_feature_classifier.classifier import spec_from_pipeline
from q2_types.feature_data import DNAIterator
from q2_types.feature_table import FeatureTable, RelativeFrequency
from q2_types.feature_data._transformer import (_read_from_fasta, _taxonomy_formats_to_dataframe, 
                                                _fastaformats_to_series)
from q2_feature_classifier._skl import _extract_reads, _specific_fitters, fit_pipeline
from q2_types.feature_data import (TSVTaxonomyFormat, HeaderlessTSVTaxonomyFormat, 
                                   FeatureData, Taxonomy, Sequence, DNAIterator, 
                                   DNAFASTAFormat)
from q2_feature_classifier.classifier import _load_class, spec_from_pipeline, pipeline_from_spec
from tax_credit.framework_functions import (
    gen_param_sweep, generate_per_method_biom_tables, move_results_to_repository)


class MyCustomTensorFlowModel:
    def __init__(self, n_units=64):
        self.n_units = n_units
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(self.n_units, activation='relu', input_shape=(input_shape,)),  # Replace input_shape
            Dense(num_classes, activation='softmax')  # Replace num_classes
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)  # Define epochs and batch_size

    def predict(self, x):
        return self.model.predict(x)

def generate_pipeline_sweep(method_parameters_combinations, reference_dbs, sweep):
    for method, params in method_parameters_combinations.items():
        classifier_params, pipeline_params = split_params(params)
        for db, refs in reference_dbs.items():
            for param_product in product(*[params[id_] for id_ in pipeline_params]):
                pipeline_param = dict(zip(pipeline_params, param_product))
                subsweep = [p for p in sweep if split_params(p[5])[1] == pipeline_param and p[2] == refs[0]]
                yield method, db, pipeline_param, subsweep

def split_params(params):
    classifier_params = feature_classifier.methods.classify_sklearn.signature.parameters.keys()
    pipeline_params = {k:v for k, v in params.items() if k not in classifier_params}
    classifier_params = {k:v for k, v in params.items() if k in classifier_params}
    return classifier_params, pipeline_params

def train_classifier(ref_reads, ref_taxa, params, pipeline, verbose=False):
    ref_reads = Artifact.load(ref_reads)
    ref_taxa = Artifact.load(ref_taxa)

    # Create an instance of your custom TensorFlow model
    custom_tf_model = MyCustomTensorFlowModel()

    # Train your TensorFlow model
    custom_tf_model.train(ref_reads, ref_taxa, params)  # You need to implement this method in your model

    # Return the trained model
    return custom_tf_model

# Modify the run_classifier function to use your custom TensorFlow model
def run_classifier(classifier, output_dir, input_dir, params, verbose=False):
    rep_seqs = Artifact.load(join(input_dir, 'rep_seqs.qza'))
    if verbose:
        print(output_dir)
    # Replace this with predictions using your TensorFlow model
    predictions = classifier.predict(rep_seqs)  # You need to implement this method in your model
    # Save the predictions to the output directory
    makedirs(output_dir, exist_ok=True)
    output_file = join(output_dir, 'taxonomy.tsv')
    dataframe = DataFrame(predictions)  # Modify this line to convert your predictions to a DataFrame
    dataframe.to_csv(output_file, sep='\t', header=False)

def train_and_run_classifier(method_parameters_combinations, reference_dbs, pipelines, sweep, verbose=False, n_jobs=4):
    for method, db, pipeline_param, subsweep in generate_pipeline_sweep(method_parameters_combinations, reference_dbs, sweep):
        ref_reads, ref_taxa = reference_dbs[db]
        classifier = train_classifier(ref_reads, ref_taxa, pipeline_param, pipelines[method], verbose=verbose)
        Parallel(n_jobs=n_jobs)(delayed(run_classifier)(
            classifier, output_dir, input_dir, split_params(params)[0], verbose=verbose) for output_dir, input_dir, rs, rt, mt, params in subsweep)
        
def main_wrapper_function(database_name, reference_seqs, reference_tax):
    analysis_name = 'mock-community'
    data_dir = join('data', analysis_name)
    precomputed_dir = join('data', 'precomputed-results', analysis_name)
    results_dir = join('temp_results_narrow')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    reference_dbs = {database_name: (reference_seqs, reference_tax)}

    dataset_reference_combinations = [
        ('mock-1', database_name),
        ('mock-2', database_name),
        ('mock-3', database_name),
        # ... (rest of the combinations)
    ]

    method_parameters_combinations = {
        'q2-TF': {
            'confidence': [0.7],
            # Add other hyperparameters relevant to your TensorFlow model here
        }
    }

    hash_params = dict(
        analyzer='char_wb', ngram_range=[8, 8], alternate_sign=False
    )

    classify_params = dict()

    pipelines = {'q2-TF': build_pipeline(MyCustomTensorFlowModel, hash_params, classify_params)}

    sweep = gen_param_sweep(
        data_dir,
        results_dir,
        reference_dbs,
        dataset_reference_combinations,
        method_parameters_combinations,
    )
    sweep = list(sweep)

    train_and_run_classifier(
        method_parameters_combinations,
        reference_dbs,
        pipelines,
        sweep,
        verbose=True,
        n_jobs=1,
    )

    taxonomy_glob = join(results_dir, 'mock-*', database_name, 'q2-TF', '*', 'taxonomy.tsv')
    generate_per_method_biom_tables(taxonomy_glob, data_dir)

    precomputed_results_dir = join("data", "precomputed-results", analysis_name)
    method_dirs = glob(join(results_dir, '*', '*', '*', '*'))
    move_results_to_repository(method_dirs, precomputed_results_dir)

def main():
    parser = argparse.ArgumentParser(description='Specify reference sequences and taxonomy.')
    parser.add_argument('-n', '--reference_database_name', nargs='?', default='gg_13_8_otus',
             help='Name of the reference database containing ref sequences and taxa. Default: %(default)s')
    parser.add_argument('-s', '--reference_sequences_path', nargs='?', type=pathlib.Path,default='data/ref_dbs/gg_13_8_otus/99_otus_v4.qza',
             help='Path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files) Default: %(default)s')
    parser.add_argument('-t', '--reference_taxonomy_path', nargs='?', type=pathlib.Path,
                        default='data/ref_dbs/gg_13_8_otus/99_otu_taxonomy_clean.tsv.qza',
             help='Path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files) Default: %(default)s')
    args = parser.parse_args()
    reference_database_name = args.reference_database_name
    reference_sequences_path = args.reference_sequences_path
    reference_taxonomy_path = args.reference_taxonomy_path
    
    print(".........debugging............")
    print(reference_database_name)
    print(reference_sequences_path)
    print(reference_taxonomy_path)
    
    main_wrapper_function(reference_database_name,reference_sequences_path,reference_taxonomy_path)

if __name__ == '__main__':
    main()
