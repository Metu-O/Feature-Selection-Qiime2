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
import pandas as pd
import numpy as np
import skbio
from Bio import SeqIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from qiime2.plugins import feature_classifier
from qiime2 import Artifact
from qiime2.plugin import Int, Str, Float, Bool, Choices, Range
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
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


class MyCustomTensorFlowModel(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.n_units = kwargs.get('units_per_layer',64)
        self.num_layers = kwargs.get('num_layers',3)
        self.batch_size = kwargs.get('batch_size',32)
        self.activation = kwargs.get('activation_function','softmax')
        self.dropout_rate = kwargs.get('dropout_rate',0.2)
        self.learning_rate = kwargs.get('learning_rate',0.01)
        self.model = None
 
    #def build_model(self, input_shape, num_classes):
    #    model = Sequential([
    #        Dense(self.n_units, activation='relu'),
    #        Flatten(),
    #        Dense(num_classes, self.activation) 
    #    ])
    #    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #    return model
    
    def build_model(self, input_shape, num_classes):
        model = tf.keras.Sequential()
        for _ in range(self.num_layers):
            model.add(tf.keras.layers.Dense(self.n_units, activation='relu'))
            model.add(tf.keras.layers.Dropout(self.dropout_rate))  # Dropout layer
           
        # Output layer (modify based on your task, e.g., binary classification, multi-class, regression)
        model.add(tf.keras.layers.Dense(num_classes, activation=self.activation)) 
       
        # Compile the model (modify based on your task)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
       
        return model
    
    def fit(self, encoded_sequences, y_train, epochs=10):
        
        # Print the parameters
        print("Model Parameters:")
        print(f"Number of Units: {self.n_units}")
        print(f"Number of Layers: {self.num_layers}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"activation_function for output layer: {self.activation}")
        
        x_train = np.stack(encoded_sequences)
        num_samples, seq_len, one_hot_dim = x_train.shape
        x_train_reshaped = x_train.reshape((num_samples, seq_len * one_hot_dim))
        input_shape = (x_train.shape[1]*x_train.shape[2])
   
        # Convert y_train to integer labels if they're not
        if isinstance(y_train[0], str):
            self.le = LabelEncoder()
            y_train = self.le.fit_transform(y_train)
 
        num_classes = len(np.unique(y_train))
   
        # Convert integer labels to one-hot encoded labels
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
        self.model = self.build_model(input_shape, num_classes)
        
        print('.............debugging.............')
        print(x_train_reshaped.shape)
        print(y_train_onehot.shape)
        
        
        #exit()
        self.model.fit(x_train_reshaped, y_train_onehot, epochs=epochs, batch_size=self.batch_size)
        return self
 
    def predict(self, x):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
            
        predictions = self.model.predict(x)
        predicted_integers = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
       
        # Convert integers back to original labels
        predicted_labels = self.le.inverse_transform(predicted_integers)
       
        return predicted_labels, confidence_scores
    
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
    # Load artifacts
    ref_reads_artifact = Artifact.load(ref_reads)
    ref_taxa_artifact = Artifact.load(ref_taxa)
    
    # 1. Parse the FASTA file
    export_dir = '/home/metu/Feature-Selection-Qiime2/data/exported/fasta/train'
    ref_reads_artifact.export_data(export_dir)
    fasta_filepath = os.path.join(export_dir, "dna-sequences.fasta")
    
    # Read sequences into a dictionary
    seq_dict = {}
    with open(fasta_filepath, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_dict[record.id] = str(record.seq)
 
    # Print first five sequences --- checking
    for idx, (id_, seq) in enumerate(seq_dict.items()):
        if idx >= 5:
            break
        print(id_, seq)
         
    print(list(seq_dict.keys())[0:5])
    
    # 2. Parse the taxonomy file
    tax_df = ref_taxa_artifact.view(pd.DataFrame)
    print(tax_df.head())
    print(tax_df.columns)
    tax_dict = tax_df['Taxon'].to_dict()
    
    print(list(tax_dict.keys())[0])
    print(seq_dict['370251'])
    overlapping_keys = [id_ for id_ in seq_dict if id_ in tax_dict]
    print(f"Number of overlapping keys: {len(overlapping_keys)}")
    
    # 3. Match sequences with taxonomies
    #matched_data = [(seq, tax_dict[id_]) for id_, seq in seq_dict.items() if id_ in tax_dict]
    #sequences, taxonomies = zip(*matched_data)
    
    matched_data = []
    for id_, seq in seq_dict.items():
        if id_ in tax_dict:
            taxonomy = tax_dict[id_]
            matched_data.append((seq, taxonomy))
    sequences, taxonomies = zip(*matched_data)
    print("These is the first mapped data",matched_data[0])
 
    # 4. Process the matched data
    encoded_sequences = [one_hot_encode(seq) for seq in sequences]
    print("These is the first encoded_sequence", encoded_sequences[0])
    print("These is the length of the first encoded_sequence", len(encoded_sequences[0]))
    # Check outer length
    all_outer_lengths_correct = all(len(seq) == 250 for seq in encoded_sequences)
 
    # Check inner lengths
    all_inner_lengths_correct = all(all(len(base_encoding) == 4 for base_encoding in seq) for seq in encoded_sequences)
 
    print(f"All outer lengths are 250: {all_outer_lengths_correct}")
    print(f"All inner lengths are 4: {all_inner_lengths_correct}")
    
    # Find sequences not of length 250
    for idx, seq in enumerate(encoded_sequences):
        if len(seq) != 250:
            print(f"Sequence at index {idx} has length {len(seq)}")
 
    # Note: You might need to further process 'taxonomies' to fit your model's output.
    # For simplicity, let's assume they are already integers representing class IDs.
    y_train = np.array(taxonomies)
 
    # Create an instance of your custom TensorFlow model
    custom_tf_model = MyCustomTensorFlowModel(**params)
 
    # 5. Train the model
    custom_tf_model.fit(encoded_sequences, y_train)
 
    # Return the trained model
    return custom_tf_model


# Modify the run_classifier function to use your custom TensorFlow model
def run_classifier(classifier, output_dir, input_dir, params, verbose=False):
    # Load the artifact
    rep_seqs_artifact = Artifact.load(join(input_dir, 'rep_seqs.qza'))
 
    # Convert the artifact to sequences (similar to what you did during training)
    export_dir = '/home/metu/Feature-Selection-Qiime2/data/exported/fasta/test'
    rep_seqs_artifact.export_data(export_dir)
    fasta_filepath = os.path.join(export_dir, "dna-sequences.fasta")
   
    seq_list = []
    with open(fasta_filepath, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_list.append(str(record.seq))
 
    # Encode the sequences
    encoded_sequences = [one_hot_encode(seq) for seq in seq_list]
   
    # Reshape the sequences for your model
    x_test = np.stack(encoded_sequences)
    num_samples, seq_len, one_hot_dim = x_test.shape
    x_test_reshaped = x_test.reshape((num_samples, seq_len * one_hot_dim))
   
    # Predict using your model
    predictions, confidence_scores = classifier.predict(x_test_reshaped)
    makedirs(output_dir, exist_ok=True)
    output_file = join(output_dir, 'taxonomy.tsv')
    dataframe = DataFrame({'PredictedLabel':predictions,
                           'Confidence':confidence_scores})  
    dataframe.to_csv(output_file, sep='\t', header=False)

def train_and_run_classifier(method_parameters_combinations, reference_dbs, pipelines, sweep, verbose=False, n_jobs=4):
    for method, db, pipeline_param, subsweep in generate_pipeline_sweep(method_parameters_combinations, reference_dbs, sweep):
        ref_reads, ref_taxa = reference_dbs[db]
        classifier = train_classifier(ref_reads, ref_taxa, pipeline_param, pipelines[method], verbose=verbose)
        Parallel(n_jobs=n_jobs)(delayed(run_classifier)(
            classifier, output_dir, input_dir, split_params(params)[0], verbose=verbose) for output_dir, input_dir, rs, rt, mt, params in subsweep)
        
def build_pipeline(model_class, hash_params, classify_params):
    
    custom_tf_model = model_class()
    
    pipeline = Pipeline([
        ('vectorizer', HashingVectorizer(**hash_params)),
        ('classifier', custom_tf_model)
    ])
    return pipeline

def one_hot_encode(sequence, desired_length=250):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded = [mapping.get(base, [0, 0, 0, 0]) for base in str(sequence)]  # use a default value for bases not in mapping
   
    # Pad the sequence if it's shorter than the desired length
    while len(encoded) < desired_length:
        encoded.append([0, 0, 0, 0])
   
    # Truncate the sequence if it's longer than the desired length
    return encoded[:desired_length]
                         
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
        'learning_rate': [0.001, 0.01],
        'batch_size': [32, 64, 128],
        'num_layers': [3, 4, 5],
        'units_per_layer': [64, 128, 256],
        'activation_function': ['softmax','sigmoid'],
        'dropout_rate': [0,0.2, 0.5]
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
