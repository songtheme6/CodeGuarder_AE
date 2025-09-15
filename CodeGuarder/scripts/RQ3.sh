#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Environment and dependency checks ---
CONDA_ENV_NAME="CodeGuarder"

# Check if 'jq' is installed for JSON processing
if ! command -v jq &> /dev/null
then
    echo "jq is not installed. Please install it to parse JSON output."
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    echo "On macOS: brew install jq"
    exit 1
fi

# Check if 'conda' is in the system's PATH
if ! command -v conda &> /dev/null
then
    echo "Conda is not found. Please ensure it is in your PATH."
    exit 1
fi

# Check if the required Conda environment exists
if ! conda info --envs | grep -q "${CONDA_ENV_NAME}"
then
    echo "Conda environment '${CONDA_ENV_NAME}' not found."
    echo "Please create it with: conda create -n ${CONDA_ENV_NAME} python=3.10"
    exit 1
fi

# --- 2. Define models, languages, and categories ---
# MODELS=("GPT-4o" "DS-V3" "CodeLlama" "DS-Coder")
#LANGUAGES=("c" "cpp" "java" "python")
# CATEGORIES=("std" "poisoning_I" "poisoning_II")

MODELS=("GPT-4o" "DS-V3" "CodeLlama" "DS-Coder")
LANGUAGES=("csharp" "javascript" "php" "rust")
CATEGORIES=("std" "poisoning_I" "poisoning_II")


# Use a temporary file to store JSON results for structured processing
TEMP_FILE=$(mktemp)

# --- 3. Execute Python scripts and capture JSON output ---
for category in "${CATEGORIES[@]}"; do
    echo "Processing category: ${category}..."
    for model in "${MODELS[@]}"; do
        echo "  Processing model: ${model}..."

        # Use 'conda run' to execute Python script in the specified environment.
        # '--no-capture-output' ensures the output is piped correctly.
        ORI_JSON=$(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python ./src/sec_eval.py --missing_kng --result_path "results/${category}_ori_${model}.json" | tail -1)
        DEF_JSON=$(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python ./src/sec_eval.py --missing_kng --result_path "results/${category}_def_${model}.json" | tail -1)

        # Store results in the temporary file, including category info
        echo "{\"category\": \"${category}\", \"model\": \"${model}\", \"ori\": ${ORI_JSON}, \"def\": ${DEF_JSON}}" >> "${TEMP_FILE}"
    done
done

# --- 4. Print Security Rate (SR) analysis table ---
echo ""
echo "--- Security Rate (SR) Analysis ---"

for category in "${CATEGORIES[@]}"; do
    echo ""
    echo "Category: ${category}"
    echo "--------------------------"
    # Print the table header
    printf "%-15s %-10s" "Model" "Type"
    for lang in "${LANGUAGES[@]}"; do
        printf "%-10s" "$lang"
    done
    printf "%-10s\n" "Average"

    # Read the data for the current category from the temp file
    grep "\"category\": \"${category}\"" "${TEMP_FILE}" | while read -r line; do
        model_name=$(echo "$line" | jq -r '.model')

        # Debugging: print the JSON line that is about to be processed
        # echo "DEBUG: Processing JSON line for model '${model_name}'" >&2
        # echo "DEBUG: JSON data: ${line}" >&2

        # Print Original (Ori) data line
        printf "%-15s %-10s" "$model_name" "Ori"
        ori_sum=0
        for lang in "${LANGUAGES[@]}"; do
            # Safely get the value, defaulting to a string "0.00" if null
            ori_rate_str=$(echo "$line" | jq -r ".ori.${lang}.pass_rate // 0")
            # Convert the string to a floating point number for arithmetic
            ori_rate=$(echo "$ori_rate_str" | awk '{printf "%.2f", $1}')
            
            # Debugging: print the extracted rate before printing
            # echo "DEBUG: Extracted original rate for ${lang}: '${ori_rate}'" >&2

            printf "%-10.2f" "$ori_rate"
            ori_sum=$(awk "BEGIN {print $ori_sum + $ori_rate}")
        done
        ori_avg=$(awk "BEGIN {print $ori_sum / ${#LANGUAGES[@]}}")
        printf "%-10.2f\n" "$ori_avg"
        
        # Print Defended (Def) data line
        printf "%-15s %-10s" "" "Def"
        def_sum=0
        for lang in "${LANGUAGES[@]}"; do
            # Safely get the value, defaulting to a string "0.00" if null
            def_rate_str=$(echo "$line" | jq -r ".def.${lang}.pass_rate // 0")
            # Convert the string to a floating point number for arithmetic
            def_rate=$(echo "$def_rate_str" | awk '{printf "%.2f", $1}')
            
            # Debugging: print the extracted rate before printing
            # echo "DEBUG: Extracted defended rate for ${lang}: '${def_rate}'" >&2

            printf "%-10.2f" "$def_rate"
            def_sum=$(awk "BEGIN {print $def_sum + $def_rate}")
        done
        def_avg=$(awk "BEGIN {print $def_sum / ${#LANGUAGES[@]}}")
        printf "%-10.2f\n" "$def_avg"
    done
done

# Clean up temporary file
rm "${TEMP_FILE}"

echo ""
echo "Analysis complete."
