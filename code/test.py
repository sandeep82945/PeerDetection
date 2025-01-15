
import os
import random
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
from pandas.core.frame import DataFrame 
import pandas as pd
import numpy  # for gradio hot reload
from watermark_processor import WatermarkLogitsProcessor_with_preferance, WatermarkDetector_with_preferance
import read_json

import torch
torch.manual_seed(123)
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

checkpoint_file = "checkpoint_new.json"
output_file = "peer_review_outputs_new.json"
run_all = True

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          OPTForCausalLM,
                          LogitsProcessorList)


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(
        description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--ppl",
        type=str2bool,
        default=False,
        help="To evaluate ppl instead of run generating and detecting",
    )
    parser.add_argument(
        "--attack_ep",
        type=float,
        default=0.1,
        help="attack epsilon. 0 for not attack",
    )
    parser.add_argument(
        "--wm_mode",
        type=str,
        default="combination",
        help="previous1 or combination",
    )
    parser.add_argument(
        "--detect_mode",
        type=str,
        default="normal",
        help="normal",
    )
    parser.add_argument(
        "--gen_mode",
        type=str,
        default="depth_d",
        help="depth_d, normal",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="sub list number",
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = 'c4',
        help = "c4, squad, xsum, PubMedQA, writingprompts",
    )
    parser.add_argument(
        "--user_dist",
        type=str,
        default="dense",
        help="sparse or dense",
    )
    parser.add_argument(
        "--user_magnitude",
        type=int,
        default=10,
        help="user number = 2**magnitude",
    )
    parser.add_argument(  
        "--delta",
        type=float,
        default=5,
        help="The (max) amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--decrease_delta",
        type=str2bool,
        default=False,
        help="Modify delta according to output length.",
        
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=25,  # 200
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=100,  
        help="Minimum number of new tokens to generate.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None, #None
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True, 
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7, #0.7
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )

    args = parser.parse_args()
    return args



def load_model(args):
    """Load and return the model and tokenizer"""

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map='auto')

    # args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]])
    # args.is_decoder_only_model = any(
    #     [(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom"]])
    # if args.is_seq2seq_model:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    # elif args.is_decoder_only_model:
    #     if args.load_fp16:
    #         model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16,
    #                                                      device_map='auto')
    #     else:
    #         model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # else:
    #     raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    # if any([(model_type in args.model_name_or_path) for model_type in ["125m", "1.3b","2.7b"]]):
    #     if args.load_fp16:
    #         pass
    #     else:
    #         model = model.to(device)
    # else:
    #     model = OPTForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto",  torch_dtype=torch.float16)
        # print(model.hf_device_map)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None,index=None, title=None):
    if title == None:
        raise ValueError("Error: 'title' cannot be None.")
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,min_new_tokens=args.min_new_tokens)

    watermark_processor = WatermarkLogitsProcessor_with_preferance(title=title,
                                                                vocab=list(tokenizer.get_vocab().values()),
                                                                args=args
                                                                )
    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))
    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )

    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        return_dict_in_generate=True, 
        output_scores=True,
        **gen_kwargs
    )

    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True, truncation=True,
                           max_length=args.prompt_max_length, return_dict=True).to(device)
    
    tokd_input = {k: v.to(model.device) for k, v in tokd_input.items()}
    
    if isinstance(tokd_input, torch.Tensor):
        tokd_input = {"input_ids": tokd_input}
    
    torch.manual_seed(args.generation_seed)
    out = generate_with_watermark(**tokd_input)
    torch.manual_seed(args.generation_seed)
    output_with_watermark = out[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)
    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    print(decoded_output_with_watermark)
    print("----------------------------------------------")
    print(decoded_output_without_watermark)
    return (tokd_input["input_ids"].shape[-1],
            output_with_watermark.shape[-1],
            redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark,
            decoded_output_with_watermark,
            watermark_processor,
            args)

def detect(input_text, args, device=None, tokenizer=None, title=None):

    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector_with_preferance(vocab=list(tokenizer.get_vocab().values()),
                                                           gamma=args.gamma,
                                                           device=device,
                                                           tokenizer=tokenizer,
                                                           z_threshold=args.detection_z_threshold,
                                                           normalizers=args.normalizers,
                                                           ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                                           select_green_tokens=args.select_green_tokens,
                                                           title=title,
                                                           args=args)
    
    score_dict, gr_score, mark = watermark_detector.detect(input_text)
    return score_dict, gr_score, mark
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        #output = list_format_scores(score_dict, watermark_detector.z_threshold)
    

    # decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    # print(decoded_output_without_watermark)

import json
import os
import numpy as np

import json
import os
import numpy as np

def safe_serialize(obj):
    """
    Helper function to safely serialize objects to JSON.
    Converts numpy objects and handles None gracefully.
    """
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None:
        return None
    return obj

def main(args):
    # Load datasets
    model, tokenizer, device = load_model(args)

    # Load existing progress if checkpoint exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_data = json.load(f)
        processed_titles = {entry["title"] for entry in processed_data}
    else:
        processed_data = []
        processed_titles = set()

    if run_all==True:
        read_file = read_json.read_old()
    else:
        read_file = read_json.data

    for each_dic in read_file:
        title = each_dic['title']
        if title in processed_titles:
            print(f"Skipping already processed: {title}")
            continue  # Skip already processed entries

        abstract = each_dic['abstract']
        paper_text = each_dic['paper_text']

        content = f''' The peer review format and length should be of standard conference.
        title: {title}
        abstract: {abstract}
        paper text: {paper_text}
        
        Steps to follow :-
        Step 1: Read the paper critically and Only write peer review and nothing else
        Step 2: In Peer review Only write Paper Summary, Strengths, Weaknesses, Suggestions for Improvement and Recommendation
        Step 2: Output Format: You must Return the Review enclosed in [Review] [\Review]
        '''

        system_prompt = ''' You are a Research Scientist. Your task is to thoroughly and critically read the paper and write a peer review of it.  Steps to follow :-
        Step 1: Read the paper critically and Only write peer review and nothing else
        Step 2: In Peer review Only write Paper Summary, Strengths, Weaknesses, Suggestions for Improvement and Recommendation
        Step 2: Output Format: You must Return the Review enclosed in [Review] [\Review]'''

        input_text = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Generate outputs
        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark, watermark_processor, _ = generate(
            input_text,
            args,
            model=model,
            device=device,
            tokenizer=tokenizer,
            title=title
        )
        
        # Detect scores for watermarked output
        if args.detect_mode == 'normal':
            gr_score_list = []
            max_sim = 0
            max_sim_idx = -1
            
            output_dict_with, gr_score_with, mark = detect(
                decoded_output_with_watermark,
                args,
                device=device,
                tokenizer=tokenizer,
                title=title
            )
            output_dict_without, gr_score_without, mark = detect(
                decoded_output_without_watermark,
                args,
                device=device,
                tokenizer=tokenizer,
                title=title
            )

        # Append the results to the list
        result = {
            "title": title,
            "abstract": abstract,
            "peer_review_without_watermark": decoded_output_without_watermark,
            "peer_review_with_watermark": decoded_output_with_watermark,
            "gr_score_with": safe_serialize(gr_score_with),  # Serialize safely
            "gr_score_without": safe_serialize(gr_score_without),  # Serialize safely
            "output_without": safe_serialize(output_dict_without),
            "output_with": safe_serialize(output_dict_with)
        }

        processed_data.append(result)
        processed_titles.add(title)

        # Save checkpoint after each iteration
        with open(checkpoint_file, "w") as f:
            json.dump(processed_data, f, indent=4)

        print(f"Processed: {title}")

    # Save the final results to the output file
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)

    # Remove the checkpoint file once done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"Saved results to {output_file}")




# def main(args):
#     # Load datasets
#     model, tokenizer, device = load_model(args)
#     output_data = []  # To store results
#     for each_dic in read_json.data:
#         title = each_dic['title']
#         abstract = each_dic['abstract']
#         paper_text = each_dic['paper_text']

#         input_text = f'''You are a Research Scientist. Your task is to thoroughly and critically read the paper and write a peer review of it.
#         title: {title}
#         abstract: {abstract}
#         paper text: {paper_text}
#         The Peer Review of the paper is :-
#         '''
#         input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark, watermark_processor,_= generate(input_text,
#                     args,
#                     model=model,
#                     device=device,
#                     tokenizer=tokenizer,
#                     title= title)
        
#         if args.detect_mode == 'normal':
                
#                     gr_score_list = []
#                     max_sim = 0
#                     max_sim_idx = -1
                    
#                     output_dict, gr_score,mark = detect(decoded_output_with_watermark,
#                                                                                     args,
#                                                                                     device=device,
#                                                                                     tokenizer=tokenizer,
#                                                                                     title=title)
#         output_data.append({
#             "title": title,
#             "abstract": abstract,
#             "peer_review_without_watermark": decoded_output_without_watermark,
#             "peer_review_with_watermark": decoded_output_with_watermark,
#             "gr_score": gr_score,
#             "mark": mark
#         })

        

#         print(gr_score)
#         exit(0)

# def main(args):
#     # Load datasets
#     model, tokenizer, device = load_model(args)
#     input_text = "How to change the car battery"
#     input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark, watermark_processor,_= generate(input_text,
#                 args,
#                 model=model,
#                 device=device,
#                 tokenizer=tokenizer,
#                 title= "my paper title")
    
#     if args.detect_mode == 'normal':
            
#                 gr_score_list = []
#                 max_sim = 0
#                 max_sim_idx = -1
                
#                 output_dict, gr_score,mark = detect(decoded_output_with_watermark,
#                                                                                 args,
#                                                                                 device=device,
#                                                                                 tokenizer=tokenizer,
#                                                                                 title="my paper title")
    


#     print(gr_score)
#     exit(0)

                                                                                    
                #     # sim = compute_similarity(mark, loop_usr_id)
                #     gr_score_list.append(gr_score)
                #     depth_score_list.append(depth_score)
                #     if gr_score> max_sim:
                #         max_sim = gr_score
                #         max_sim_idx = j
                
                # detect_range=len(usr_list)//10
                # mapped_gr_score=gr_score_list[gen_id]
                # mapped_depth_score=depth_score_list[gen_id]
                # # sim_result=np.zeros([2,detect_range])
                # gr_result=[]
                # depth_result=[]
                # id_result=[]
                # id_index=[]
             
                    
                    
                # if args.gen_mode=='normal':
                #     for r in range(detect_range): # sort with depth
                #         sim = max(gr_score_list)
                #         index = gr_score_list.index(sim)
                #         gr_score_list[index] = -100
                #         gr_result.append(sim)
                #         id_result.append(usr_list[index])
                #         id_index.append(index)

                #     result_dic={"gr_score":gr_result[:10], "id":id_result[:10]}
                #     if_succ_top1=0
                #     if_succ_top3=0
                #     if_succ_top10=0

                # if gen_id in id_index[:3]:
                #     succ_num_top3 += 1
                #     if_succ_top3=1

                # if gen_id in id_index[:1]:
                #     succ_num_top1+=1
                #     if_succ_top1=1
                    
                # if gen_id in id_index[:10]:
                #     succ_num_top10+=1
                #     if_succ_top10=1
            
            

if __name__ == "__main__":
    args = parse_args()
    # print(args)
    if args.ppl:
        testppl(args)
    else:
        main(args)
