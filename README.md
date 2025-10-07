<div align="center">

<h1>
  <img src="./assets/problem-solving.png" alt="logo" width="60" style="vertical-align:middle; margin-right:10px;">
  R-HORIZON
</h1>

<div>
   How Far Can Your Large Reasoning Model Really Go in Breadth and Depth? 
</div>
</div>

<br>


<p align="center">
  📃 <a href="TBD" target="_blank">Paper</a >  • 🤗 <a href="TBD" target="_blank">Dataset</a > 
</p >

R-HORIZON is a novel method designed to stimulate long-horizon reasoning behaviors in Large Reasoning Models (LRMs) through query composition. We transform isolated problems into complex multi-step reasoning scenarios, revealing that even the most advanced LRMs suffer significant performance degradation when facing interdependent problems that span long reasoning horizons.

![](./assets/mainfig.png)

## 🔥 Releases

**[2025-10-09]**
- 🎉 **R-HORIZON Benchmark** is now available! Test your LRMs on complex multi-horizon reasoning tasks.
- 🤗 **Training and evaluation datasets** are available on Hugging Face: [R-HORIZON Dataset](TBD)
- 📄 **Paper released** on arXiv: [R-HORIZON: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth?](TBD)


## 🌟 Overview

Recent advances in reasoning-focused language models (e.g., OpenAI o1, DeepSeek-R1) have demonstrated remarkable improvements through test-time scaling and long Chain-of-Thought (CoT). However, existing benchmarks primarily focus on immediate, single-horizon tasks, failing to adequately evaluate models' ability to handle complex, long-horizon scenarios.

**Key challenges in current paradigms:**
- **Limited evaluation scope**: Existing benchmarks confine themselves to isolated problems, missing the complexity of real-world multi-step reasoning
- **Limited effective reasoning length**: Models struggle to maintain performance as reasoning chains grow longer
- **Poor thinking budget allocation**: LRMs fail to appropriately distribute thinking resources across multiple interdependent problems

To address these limitations, we introduce **R-HORIZON**, which:
- Transforms isolated problems into **complex multi-step reasoning scenarios** through query composition
- Establishes the **R-HORIZON Benchmark** comprising 6 representative datasets from mathematics, code generation, and agent applications
- Enables **reinforcement learning with verified rewards (RLVR)** using long-horizon reasoning data

![](./assets/method_fig.png)

## 📖 Table of Contents (TODO: update)

- [R-HORIZON](#r-horizon)
  * [🔥 Releases](#-releases)
  * [🌟 Overview](#-overview)
  * [📊 R-HORIZON Benchmark](#-r-horizon-benchmark)
  * [🚀 Training with R-HORIZON](#-training-with-r-horizon)
  * [Quick Start](#quick-start)
    + [Installation](#installation)
    + [Benchmark Evaluation](#benchmark-evaluation)
    + [Training with RLVR](#training-with-rlvr)
  * [Dataset](#dataset)
  * [Citation](#citation)

## 📊 R-HORIZON Benchmark

We evaluate 20+ state-of-the-art LRMs on the R-HORIZON Benchmark, revealing significant performance degradation as reasoning horizons increase:

![](./assets/result_fig.png)

**Key findings from our benchmark evaluation:**

- **Universal performance degradation**: Even the most powerful models suffer severe drops as problem count increases. For instance, DeepSeek-R1 drops from 87.3% (single problem) to 24.6% (5 problems) on AIME25.

- **Model size matters**: Larger models exhibit more resilience to multi-horizon challenges. R1-Qwen-7B drops from 93.6% to 0% when solving 16 problems, showing 34.1% more degradation than the 32B models.

- **Task-dependent degradation**: Code generation tasks show steeper performance declines compared to mathematics. Many reasoning models lose their tool-calling abilities in web search scenarios, resulting in poor multi-step performance.

## 🚀 Training with R-HORIZON

Training with R-HORIZON composed data yields substantial improvements on both single and multi-horizon reasoning tasks:

![](./assets/skywork_n1_n2_comparison.png)

**Training results highlights:**

- **Dual Performance Gains**: Training with 2-composed problems significantly improves both multi-horizon reasoning (+17.4 points on AIME24 n=2) and single-problem performance (+7.5 points on AIME24 original).

- **Scalable Complexity**: Increasing composition complexity (n=4) enhances the model's ability to handle problems requiring more reasoning steps, achieving 50.6% on Math500 (n=8).

| Models | MATH500 (Origin) | MATH500 (n=8) | AIME24 (Origin) | AIME24 (n=2) | AIME25 (Origin) | AIME25 (n=2) | AMC23 (Origin) | AMC23 (n=2) |
|-----------------|------------------|---------------|-----------------|--------------|-----------------|--------------|----------------|-------------|
| R1-Qwen-7B | 93.6 | 11.8 | 48.3 | 16.4 | 33.3 | 3.5 | 90.2 | 48.8 |
| Baseline (n=1) | 95.6 | 8.4 | 57.9 | 16.7 | 47.9 | 5.1 | **95.9** | 55.0 |
| R-HORIZON (n=2) | 95.4 | 21.4 | **65.4** | 34.1 | **49.6** | 10.0 | 94.1 | 80.6 |
| R-HORIZON (n=4) | 94.6 | **50.6** | 62.9 | 34.8 | 45.4 | 8.1 | 91.9 | 79.1 |


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/R-HORIZON.git
cd R-HORIZON

# Create conda environment
conda create -n r-horizon python=3.10 -y
conda activate r-horizon

# Install PyTorch
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

# Install additional dependencies
pip install -r requirements.txt
```

### Benchmark Evaluation (TODO: linsen)
1. Download the R-HORIZON Benchmark

```bash
# Download benchmark datasets
python ./evaluation/data/download.py
```

2. Evaluate your model 

```python

```

### Training with R-HORIZON datasets
1. Download composed training data

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lulululuyi/R-HORIZON-training-data",
    repo_type="dataset",
    local_dir="./training/data",
)
```

2. Launch training

```bash
# Train with R-HORIZON using GRPO algorithm
bash ./training/scripts/train/skywork-or1-rlvr-math-training-7b-40k.sh
```



# Dataset
## Dataset Construction
Step 1: Filter Samples with Valid Integers  
```bash
# Purpose: Retain samples containing valid integers in input text and pure integer targets, excluding ambiguous numeric expressions (e.g., floats, fractions, LaTeX commands).  
python step1_filt_integer_samples.py
```



Step 2: Identify Key Variables
```bash
# Purpose: select "key variables" (critical integers that significantly affect problem outcomes)
# configure API credentials in the script (replace YOUR_API_KEY)
python step2_select_key_variable.py
```

Step 3: Combine into Chained Reasoning Problems
```bash
# Purpose: Generate multi-horizon chained problems where each step’s key variable depends on the previous step’s answer.
python step3_combine_problems.py
```

## Dataset on Hugging Face Hub
The R-HORIZON training datasets and evaluation benchmark are available on Hugging Face Hub:
## Dataset on Hugging Face Hub
The R-HORIZON training datasets and evaluation benchmark are available on Hugging Face Hub:
| Dataset Type | Dataset Name                  | Hugging Face Link                                                                 |
|--------------|-------------------------------|-----------------------------------------------------------------------------------|
| Evaluation   | R-HORIZON-Math500             | [link](https://huggingface.co/datasets/lulululuyi/R-HORIZON-Math500)             |
| Evaluation   | R-HORIZON-AIME24              | [link](https://huggingface.co/datasets/lulululuyi/R-HORIZON-AIME24)               |
| Evaluation   | R-HORIZON-AIME25              | [link](https://huggingface.co/datasets/lulululuyi/R-HORIZON-AIME25)               |
| Evaluation   | R-HORIZON-AMC23              | [link](https://huggingface.co/datasets/lulululuyi/R-HORIZON-AMC23)               |
| Evaluation   | R-HORIZON-Websearch           | [link](https://huggingface.co/datasets/lulululuyi/R-HORIZON-Websearch)            |
| Training     | R-HORIZON-training-data       | [link](https://huggingface.co/datasets/lulululuyi/R-HORIZON-training-data)        |



## Dataset Structure

```json
{
  "input": "[1-N linked problems + solving instructions (with [variablek]/[answerk] placeholders)]",
  "instanceId": "[Unique ID for this instance]",
  "origin_instanceIds": "[List of original problem IDs]",
  "target": "[List of final answers, e.g., [answer1, answer2]]",
  "num_problems": "[Total problems, e.g., 2]",
  "selected_variables": [
    {
      "number": "[Key variable from problem]",
      "context": "[Context of the number]",
      "text": "[Text of the number]",
      "is_independent": "[true/false]",
      "is_in_math_env": "[true/false]"
    }
    // Add/remove variables as needed
  ]
}
```

# Citation
If you find R-HORIZON helpful for your research, please cite our paper.
