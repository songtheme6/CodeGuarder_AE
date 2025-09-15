This repository contains the source code and dataset associated with our submission to the **32nd ACM Conference on Computer and Communications Security (CCS) 2025**. The paper is titled "Give LLMs a Security Course: Securing Retrieval-Augmented Code Generation via Knowledge Injection". Below, you'll find instructions for installing dependencies, setting up the environment, and running the provided tools to reproduce our results.

## Abstract
Retrieval-Augmented Code Generation (RACG) improves code correctness using external knowledge but is vulnerable to security risks, especially when its knowledge base is poisoned with malicious code. To mitigate this, we introduce CodeGuarder, a framework that hardens LLMs by integrating security knowledge directly into the code generation process.

CodeGuarder builds a security knowledge base from real-world vulnerabilities and uses a novel mechanism to retrieve, re-rank, and inject the most relevant security guidance into the LLM's prompt. Our evaluation shows that CodeGuarder significantly enhances code security under both standard and poisoning scenarios without compromising functional correctness. It also demonstrates strong generalization to new contexts, marking a key step toward building more secure and trustworthy RACG systems.


## Installation and Requirements

To evaluate the effectiveness of **CodeGuarder**, please first install the environment by running the script provided:

```bash
./scripts/init_env.sh
```

This will create a conda environment named `CodeGuarder` with all the necessary dependencies.

## Running CodeGuarder

To reproduce the experiments for **CodeGuarder**, ensure that your Large Language Model (LLM) API key is correctly set in the `config.py` file.

For **DeepSeek-Coder** and **CodeLlama**, you can follow the instructions on the [Ollama website](https://ollama.com) to deploy the models locally. Alternatively, you can leverage other online providers to invoke the LLMs.

We also provide the generated responses that can be evaluated offline to save time in artifact evaluation.

## Reproducing the Results

To reproduce the results reported in the paper, use the following commands.

### Evaluating CodeGuarder Under the Standard Scenario (RQ1)

This section evaluates the performance of CodeGuarder in a standard, non-adversarial setting.

#### Generating the result from scratch (optional)

**Without CodeGuarder (Baseline):**

```bash
./scripts/run_std_baseline.sh <MODEL_NAME> <MODEL_KEY> <BASE_URL>
```

For example:
```bash
./scripts/run_combined_benchmark.sh "openrouter/openai/gpt-4o" "YOUR_OPENROUTER_KEY" "https://openrouter.ai/api/v1/chat/completions"
```

**With CodeGuarder:**

```bash
./scripts/run_std_codeguarder.sh <MODEL_NAME> <MODEL_KEY> <BASE_URL>
```

#### Evaluating the result directly

To evaluate the pre-generated results:

```bash
./scripts/RQ1.sh
```

The output should be similar to the following:

```
--- Security Rate (SR) Analysis ---

Model           Type      c         cpp       java      python    Average   
GPT-4o          Ori       55.95     80.31     41.92     70.09     62.07     
                Def       70.04     84.94     62.45     81.20     74.66     
DS-V3           Ori       55.07     74.90     40.61     68.95     59.88     
                Def       72.25     88.03     73.36     79.49     78.28     
CodeLlama       Ori       52.42     74.13     44.98     68.66     60.05     
                Def       66.52     80.31     57.21     74.93     69.74     
DS-Coder        Ori       51.98     75.68     46.29     71.51     61.36     
                Def       65.20     80.31     55.02     78.06     69.65     

--- CodeBLEU (Sim) Analysis ---

Model           Type      c         cpp       java      python    Average   
GPT-4o          Ori       22.82     24.47     28.56     21.33     24.30     
                Def       23.71     25.46     29.15     23.36     25.42     
DS-V3           Ori       22.79     24.13     28.93     21.08     24.23     
                Def       24.07     25.41     29.41     23.45     25.59     
CodeLlama       Ori       23.19     24.29     28.09     21.72     24.32     
                Def       23.86     25.65     28.35     22.91     25.19     
DS-Coder        Ori       21.97     23.94     26.95     21.01     23.47     
                Def       22.74     24.98     27.02     22.29     24.26  
```


**Note:** The results might differ slightly from those reported in the paper due to different rounding methods (usually less than 0.1).

### Evaluating CodeGuarder Under Poisoning Scenarios (RQ2)

This section assesses the robustness of CodeGuarder against two different data poisoning scenarios.

#### Scenario I: Generating the result from scratch (optional)

**Without CodeGuarder (Baseline):**

```bash
./scripts/run_poisoning_I_baseline.sh <MODEL_NAME> <MODEL_KEY> <BASE_URL>
```

**With CodeGuarder:**

```bash
./scripts/run_poisoning_I_codeguarder.sh <MODEL_NAME> <MODEL_KEY> <BASE_URL>
```

#### Scenario I: Evaluating the result directly

```bash
./scripts/RQ2_Scenario_I.sh
```

#### Scenario II: Generating the result from scratch (optional)

**Without CodeGuarder (Baseline):**

```bash
./scripts/run_poisoning_II_baseline.sh <MODEL_NAME> <MODEL_KEY> <BASE_URL>
```

**With CodeGuarder:**

```bash
./scripts/run_poisoning_II_codeguarder.sh <MODEL_NAME> <MODEL_KEY> <BASE_URL>
```

#### Scenario II: Evaluating the result directly

```bash
./scripts/RQ2_Scenario_II.sh
```

### Evaluating CodeGuarder's Generalization (RQ3)

This section evaluates the generalization capabilities of CodeGuarder. The results for this research question are generated during the execution of the RQ2 experiments.

#### Evaluating the result directly

This evaluation will take approximately 30 minutes.

```bash
./scripts/RQ3.sh
```

