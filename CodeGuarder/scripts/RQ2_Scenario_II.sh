#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e


CONDA_ENV_NAME="CodeGuarder"

if ! command -v jq &> /dev/null
then
    echo "jq is not installed. Please install it to parse JSON output."
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    echo "On macOS: brew install jq"
    exit 1
fi

if ! command -v conda &> /dev/null
then
    echo "Conda is not found. Please ensure it is in your PATH."
    exit 1
fi

if ! conda info --envs | grep -q "${CONDA_ENV_NAME}"
then
    echo "Conda environment '${CONDA_ENV_NAME}' not found."
    echo "Please create it with: conda create -n ${CONDA_ENV_NAME} python=3.10"
    exit 1
fi

MODELS=("GPT-4o" "DS-V3" "CodeLlama" "DS-Coder")
LANGUAGES=("c" "cpp" "java" "python")

TEMP_FILE=$(mktemp)

for model in "${MODELS[@]}"; do
    echo "Processing ${model}..."

    ORI_JSON=$(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python ./src/sec_eval.py --result_path "results/poisoning_II_ori_${model}.json" | tail -1)
    DEF_JSON=$(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python ./src/sec_eval.py --result_path "results/poisoning_II_def_${model}.json" | tail -1)

    echo "{\"model\": \"${model}\", \"ori\": ${ORI_JSON}, \"def\": ${DEF_JSON}}" >> "${TEMP_FILE}"
done


echo ""
echo "--- Security Rate (SR) Analysis ---"
echo ""
printf "%-15s %-10s" "Model" "Type"
for lang in "${LANGUAGES[@]}"; do
    printf "%-10s" "$lang"
done
printf "%-10s\n" "Average"

while read -r line; do
    model_name=$(echo "$line" | jq -r '.model')

    printf "%-15s %-10s" "$model_name" "Ori"
    ori_sum=0
    for lang in "${LANGUAGES[@]}"; do
        ori_rate=$(echo "$line" | jq -r ".ori.${lang}.pass_rate")
        printf "%-10.2f" "$ori_rate"
        ori_sum=$(awk "BEGIN {print $ori_sum + $ori_rate}")
    done
    ori_avg=$(awk "BEGIN {print $ori_sum / ${#LANGUAGES[@]}}")
    printf "%-10.2f\n" "$ori_avg"
    
    printf "%-15s %-10s" "" "Def"
    def_sum=0
    for lang in "${LANGUAGES[@]}"; do
        def_rate=$(echo "$line" | jq -r ".def.${lang}.pass_rate")
        printf "%-10.2f" "$def_rate"
        def_sum=$(awk "BEGIN {print $def_sum + $def_rate}")
    done
    def_avg=$(awk "BEGIN {print $def_sum / ${#LANGUAGES[@]}}")
    printf "%-10.2f\n" "$def_avg"
done < "${TEMP_FILE}"


echo ""
echo "--- CodeBLEU (Sim) Analysis ---"
echo ""
printf "%-15s %-10s" "Model" "Type"
for lang in "${LANGUAGES[@]}"; do
    printf "%-10s" "$lang"
done
printf "%-10s\n" "Average"

while read -r line; do
    model_name=$(echo "$line" | jq -r '.model')
    
    printf "%-15s %-10s" "$model_name" "Ori"
    ori_sum=0
    for lang in "${LANGUAGES[@]}"; do
        ori_bleu=$(echo "$line" | jq -r ".ori.${lang}.code_bleu")
        printf "%-10.2f" "$ori_bleu"
        ori_sum=$(awk "BEGIN {print $ori_sum + $ori_bleu}")
    done
    ori_avg=$(awk "BEGIN {print $ori_sum / ${#LANGUAGES[@]}}")
    printf "%-10.2f\n" "$ori_avg"
    
    printf "%-15s %-10s" "" "Def"
    def_sum=0
    for lang in "${LANGUAGES[@]}"; do
        def_bleu=$(echo "$line" | jq -r ".def.${lang}.code_bleu")
        printf "%-10.2f" "$def_bleu"
        def_sum=$(awk "BEGIN {print $def_sum + $def_bleu}")
    done
    def_avg=$(awk "BEGIN {print $def_sum / ${#LANGUAGES[@]}}")
    printf "%-10.2f\n" "$def_avg"
done < "${TEMP_FILE}"

# Clean up temporary file
rm "${TEMP_FILE}"

echo ""
echo "Analysis complete."
