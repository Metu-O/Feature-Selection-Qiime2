#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=30Gb
#SBATCH --output=script.out
#SBATCH --time=24:00:00
 
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/SE/BMIG-6202-MSR/qiime2-2022.8 #change to the environment where qiime2-2022.8 is downloaded.

wd="/scratch/metu/input/" #temporary working directory. Change to your own path.
if [ ! -d "$wd" ]; then
  mkdir -p "$wd"
else
  rm -rf "$wd"/*
fi
echo "working directory created/emptied"

#copy the necessary codes and data to your temporary directory
cp ~/Feature-Selection-Qiime2/Naive_Bayes_Parameters.py "$wd"
cp ~/Feature-Selection-Qiime2/SelectFromModel_MultinomialNB.py "$wd"
cp ~/Feature-Selection-Qiime2/SelectFromModel_RandomForest.py "$wd"
cp ~/Feature-Selection-Qiime2/SelectFromModel_SGD.py "$wd"
cp ~/Feature-Selection-Qiime2/Evaluate_Method_Accuracy.py "$wd"
cp -r ~/Feature-Selection-Qiime2/tax_credit "$wd"
cp -r ~/Feature-Selection-Qiime2/data "$wd"

#change directory to working directory
cd "$wd"

format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }

python Naive_Bayes_Parameters.py #running this line uses the default input

#run these lines to use your own reference database
#python Naive_Bayes_Parameters.py \
#   -n 'reference database name'\
#   -s 'path to reference sequences. QIIME2 ARTIFACTS ONLY (.qza files)'\
#   -t 'path to reference taxonomy. QIIME2 ARTIFACTS ONLY (.qza files)'

#see example below
#python Naive_Bayes_Parameters.py \
#   -n 'greegenes'\
#   -s '/home/mosele/99_gg_seq.qza'\
#   -t '/home/mosele/99_otu_taxonomy_clean-Copy1.tsv.qza'

echo "Naive_Bayes_Parameters.py script completed after $(format_time $SECONDS)"

python SelectFromModel_MultinomialNB.py #running this line uses the default input
#run these lines to use your own reference database
#python SelectFromModel_MultinomialNB.py \
#   -n 'reference database name'\
#   -s 'path to reference sequences'\
#   -t 'path to reference taxonomy'

echo "SelectFromModel_MultinomialNB.py script completed after $(format_time $SECONDS)"

python SelectFromModel_RandomForest.py #running this line uses the default input
#run these lines to use your own reference database
#python SelectFromModel_RandomForest.py \
#   -n 'reference database name'\
#   -s 'path to reference sequences'\
#   -t 'path to reference taxonomy'


echo "SelectFromModel_RandomForest.py script completed after $(format_time $SECONDS)"

python SelectFromModel_SGD.py #running this line uses the default input
#run these lines to use your own reference database
#python SelectFromModel_SGD.py \
#   -n 'reference database name'\
#   -s 'path to reference sequences'\
#   -t 'path to reference taxonomy' 

echo "SelectFromModel_SGD.py script completed after $(format_time $SECONDS)"

python Evaluate_Method_Accuracy.py #running this line uses the default input
#run these lines to chose your own output path
#python Evaluate_Method_Accuracy.py \
#   -r 'name of database containing ref sequences and taxa e.g gg_13_8_otus, greengenes, SILVA etc.'
#   -p 'save plots in this directory'

#see example below
#python Evaluate_Method_Accuracy.py \
#   -r 'greegenes'
#   -p '/home/mosele/plots'

echo "Evaluate_Method_Accuracy.py script completed after $(format_time $SECONDS)"
echo "That means this whole script took $(format_time $SECONDS) to run"
echo "I'm Done"