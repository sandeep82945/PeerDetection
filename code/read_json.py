import pandas as pd
import json
with open('/Users/sandeepkumar/Projects/source_watermarking/try_1.json') as f:
    data = json.load(f)


# print(data[0]["title"])
# import torch, 

# def _seed_rng(input_ids: torch.LongTensor, seeding_scheme: str = "simple_1") -> torch.Generator:
#     hash_key: int = 15485863
#     rng = torch.Generator(device='cpu')  # Explicitly set to CPU

#     if seeding_scheme == "simple_1":
#         assert input_ids.shape[-1] >= 1, (
#             f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed RNG"
#         )
#         prev_token = input_ids[-1].item()
#         print("Previous token for RNG seed:", prev_token)
#         rng.manual_seed(hash_key * prev_token)
#     else:
#         raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")

#     return rng


# if __name__ == "__main__":
#     # Example input_ids
#     input_ids = torch.LongTensor([101, 202, 303])  # Simulated token IDs

#     # Call _seed_rng
#     rng = _seed_rng(input_ids)

#     # Generate some random numbers with the seeded generator
#     random_numbers = [torch.rand(1, generator=rng).item() for _ in range(5)]
#     print("\nRandom numbers after seeding:", random_numbers)



# # Create a random generator and set a seed
# rng = torch.Generator(device='cpu')
# rng.manual_seed(42)

# # Set vocabulary size
# vocab_size = 10

# # Generate random permutation
# vocab_permutation = torch.randperm(vocab_size, device='cpu', generator=rng)
# print("Random Permutation of Vocabulary Indices:", vocab_permutation)

# # Simulate greenlist and redlist
# gamma = 0.5  # 50% green tokens
# greenlist_size = int(vocab_size * gamma)
# greenlist_ids = vocab_permutation[:greenlist_size]
# redlist_ids = vocab_permutation[greenlist_size:]


# print("Greenlist IDs:", greenlist_ids)
# print("Redlist IDs:", redlist_ids)
