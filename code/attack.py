from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import random
import torch
from functools import partial

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to replace one token based on logits
def replace_1_token(cut_prompt):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cut_prompt = cut_prompt.unsqueeze(0)
    generate_with_score = partial(
        model.generate,
        output_scores=True,
        return_dict_in_generate=True,
    )
    out = generate_with_score(cut_prompt, pad_token_id=tokenizer.pad_token_id, max_new_tokens=1)
    res = out[0][0][-1]
    logits = torch.stack(out[1])
    sm = nn.Softmax(dim=1)
    probabilities = sm(logits.squeeze(0))
    top_values, top_indices = torch.topk(probabilities, 2)

    top1ind = top_indices[0][0].item()
    top2ind = top_indices[0][1].item()
    return res, top1ind, top2ind

# Function to process the attack
def attack_process(decoded_output, epsilon):
    skip = False
    tokd_input = tokenizer(decoded_output, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=300)["input_ids"].to(device)
    input_length = tokd_input.shape[-1]
    prefix_length = 31  # Fixed prefix length to protect initial tokens

    if input_length <= prefix_length:
        print(f"Skipping: Input too short ({input_length} tokens).")
        return decoded_output, True

    attack_num = int(min(epsilon * input_length, input_length - prefix_length))
    if attack_num <= 0:
        print("Skipping: Not enough tokens to attack.")
        return decoded_output, True

    random_indices = random.sample(range(prefix_length, input_length - 1), attack_num)
    attacked_output = tokd_input.clone()

    for rand_idx in random_indices:
        cut_prompt = tokd_input[0][rand_idx - prefix_length:rand_idx]
        standard_token = tokd_input[0][rand_idx]
        select_token, top1ind, top2ind = replace_1_token(cut_prompt)

        if standard_token != select_token:
            attacked_output[0][rand_idx] = select_token
        else:
            if standard_token != top1ind:
                attacked_output[0][rand_idx] = top1ind
            else:
                attacked_output[0][rand_idx] = top2ind

    decoded_output = tokenizer.decode(attacked_output[0], skip_special_tokens=True)
    return decoded_output, skip



def paraphrase_local_gpt(prompt, level=1):
    # Load tokenizer and model (e.g., GPT-2)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    instructions = {
        1: "Paraphrase the following text with minimal changes. Keep the meaning as close to the original as possible:",
        2: "Paraphrase the following text moderately. Change some sentence structures and wording but keep the overall meaning:",
        3: "Paraphrase the following text significantly. Rephrase it completely while maintaining the meaning:",
    }

    instruction = instructions.get(level, instructions[1])
    full_prompt = f"{instruction}\n\nText: {prompt}\n\nParaphrased Version:"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=200, temperature=0.7)
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return paraphrased_text


# Example usage
prompt = "The systems interconnect is expected to cost"
print(paraphrase_local_gpt(prompt))
# decoded_output = prompt
# epsilon = 0.2  # Fraction of tokens to attack

# # Perform attack
# attacked_output, skip = attack_process(decoded_output, epsilon)
# if not skip:
#     print("Attacked Output:", attacked_output)
# else:
#     print("Attack skipped.")
