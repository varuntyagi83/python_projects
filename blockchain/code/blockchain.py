import hashlib
import datetime
import numpy as np

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

def calculate_hash(index, previous_hash, timestamp, data):
    value = f"{index}{previous_hash}{timestamp}{data}"
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def create_genesis_block():
    # Genesis block has index 0 and arbitrary previous hash
    return Block(0, "0", datetime.datetime.now(), "Genesis Block", calculate_hash(0, "0", datetime.datetime.now(), "Genesis Block"))

def create_new_block(previous_block, data):
    index = previous_block.index + 1
    timestamp = datetime.datetime.now()
    hash = calculate_hash(index, previous_block.hash, timestamp, data)
    return Block(index, previous_block.hash, timestamp, data, hash)

# Create the blockchain
blockchain = [create_genesis_block()]
previous_block = blockchain[0]

# Add blocks to the blockchain
num_blocks = 5
for i in range(1, num_blocks + 1):
    new_data = f"Block {i} Data"
    new_block = create_new_block(previous_block, new_data)
    blockchain.append(new_block)
    previous_block = new_block

# Display the blockchain
for block in blockchain:
    print(f"Block #{block.index} | Timestamp: {block.timestamp} | Data: {block.data} | Hash: {block.hash}")
    print("-" * 50)
