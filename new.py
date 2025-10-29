def wha_hash_bytes(input_bytes: bytes) -> int:
    mask = 0x3FFFFFFF
    outHash = 0

    for byte in input_bytes:
        intermediate_value = (
            ((byte ^ 0xCC) << 24) |
            ((byte ^ 0x33) << 16) |
            ((byte ^ 0xAA) << 8) |
            (byte ^ 0x55)
        )
        outHash = (outHash & mask) + (intermediate_value & mask)

    return outHash

# Target hash from known input string
input_string = "IN ORDER TO PREVENT CONFLICT YOU MIGHT PASS THIS SACRED CEREMONIAL OBJECT USED BY SOME NATIVE AMERICANS"
input_hash = wha_hash_bytes(input_string.encode("utf-8"))

# Brute force search
from tqdm import tqdm

for i in tqdm(range(2**30)):
    b = i.to_bytes((i.bit_length() + 7) // 8 or 1, 'big')
    if wha_hash_bytes(b) == input_hash:
        print("Collision found:", b.hex())
        break
