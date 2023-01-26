

dataset=movie-lens_full

mkdir -p data/${dataset}
python -u trans_gwf_data.py ../HGB/Data/${dataset}/train.txt data/${dataset}/train.txt 1
python -u trans_gwf_data.py ../HGB/Data/${dataset}/test.txt data/${dataset}/test.txt 0
awk -F " " '{print $1"\t"$2"\t"$3}' ../HGB/Data/${dataset}/kg_final.txt > data/${dataset}/kg_final.txt
