# Step1 Inference
python inference.py --input combined_key_var_k2_pretty_formatted.jsonl --output r1distillqwen7b-combined_key_var_k2_pretty_formatted_result.jsonl --model_name r1-distill-qwen7b
# Step2 use llm to extract nested answer from inference result
python extract.py --input r1distillqwen7b-combined_key_var_k2_pretty_formatted_result.jsonl --output r1distillqwen7b-combined_key_var_k2_pretty_formatted_result_judged.jsonl --model_name gpt-4.1
# Step3 use verify script to judge
python judge.py --raw_input combined_key_var_k2_pretty_formatted.jsonl --prediction r1distillqwen7b-combined_key_var_k2_pretty_formatted_result_judged.jsonl --output r1distillqwen7b-combined_key_var_k2_pretty_formatted_result_stat.txt