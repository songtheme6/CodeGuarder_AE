#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Utility Functions ---
# Check if a command exists
command_exists () {
    command -v "$1" >/dev/null 2>&1
}

# --- Check Prerequisites ---
if ! command_exists jq; then
    echo "jq is not installed. Please install it to parse JSON output."
    echo "On Debian/Ubuntu: sudo apt-get install jq"
    echo "On macOS: brew install jq"
    exit 1
fi

# --- Parameter Validation ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <MODEL_NAME> <MODEL_KEY> <BASE_URL>"
    exit 1
fi

MODEL_NAME="$1"
MODEL_KEY="$2"
BASE_URL="$3"


# Other configs
CONDA_ENV_NAME="CodeGuarder"
CG_PYTHON_SCRIPT="/home/bo/Security/CodeGuarder/src/Defense_Poisoning_I.py"
OUTPUT_PATH="./dataset/Poisoning_I_Def.json"
ORI_INSTRUCTION_PATH="./dataset/Instruction.json"
KNG_BASE_PATH="./dataset/Root_Causes.json"
BENCHMARK_MODULE_TO_RUN="CybersecurityBenchmarks.benchmark.run"
RESPONSE_PATH="./results/poisoning_I_def_${MODEL_NAME}.json"
BROKEN_INST_PATH="./dataset/Broken_instruct.json"
POISONED_PATH="./dataset/Poisoning_I.json"

# --- Main Script Execution ---

echo "--- Step 1: Running CodeGuarder script in '${CONDA_ENV_NAME}' environment ---"

# Execute the first Python script
conda run --no-capture-output -n "${CONDA_ENV_NAME}" python "${CG_PYTHON_SCRIPT}" \
    --output_path "${OUTPUT_PATH}" \
    --ori_instruction_path "${ORI_INSTRUCTION_PATH}" \
    --root_cause_path "${KNG_BASE_PATH}" \
    --broken_instruction_path "${BROKEN_INST_PATH}" \
    --poisoned_instructions_path "${POISONED_PATH}" \

# Check if the first command succeeded
if [ $? -ne 0 ]; then
    echo "Error: The CodeGuarder script failed. Exiting."
    exit 1
fi

# echo "--- Step 2: Running CybersecurityBenchmarks script in '${CONDA_ENV_NAME}' environment ---"

# Construct the formatted LLM string
LLM_TEST_STRING="OPENAI::${MODEL_NAME}::${MODEL_KEY}::${BASE_URL}"

# Execute the second Python script, explicitly setting the working directory
conda run --no-capture-output -n "${CONDA_ENV_NAME}" python -m "${BENCHMARK_MODULE_TO_RUN}" \
    --benchmark=instruct \
    --prompt-path="${OUTPUT_PATH}" \
    --response-path="${RESPONSE_PATH}" \
    --llm-under-test="${LLM_TEST_STRING}" \


echo "--- Step 3: Running evaluation script and parsing results ---"

# Run the sec_eval.py script and capture the last line (JSON output)
EVAL_JSON=$(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python ./src/sec_eval.py --result_path "${RESPONSE_PATH}" | tail -1)

# Check if the output is a valid JSON
if ! echo "${EVAL_JSON}" | jq -e . >/dev/null 2>&1; then
    echo "Error: The evaluation script did not produce valid JSON output."
    echo "Output was: ${EVAL_JSON}"
    exit 1
fi

# Print formatted output for each language
echo ""
echo "--- Final Results from Evaluation ---"
echo ""
printf "%-10s %-10s %-10s\n" "Language" "SR" "CodeBLEU"
echo "-----------------------------------"

# Use the 'keys' filter which is widely supported
LANGUAGES=$(echo "${EVAL_JSON}" | jq -r 'keys | .[]')
for lang in ${LANGUAGES}; do
    SR_VALUE=$(echo "${EVAL_JSON}" | jq -r ".${lang}.pass_rate")
    CODEBLEU_VALUE=$(echo "${EVAL_JSON}" | jq -r ".${lang}.code_bleu")
    printf "%-10s %-10.2f %-10.2f\n" "$lang" "$SR_VALUE" "$CODEBLEU_VALUE"
done

echo ""
echo "--- Script finished ---"