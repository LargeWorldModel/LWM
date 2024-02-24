import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='LargeWorldModel/LWM-Text-Chat-256K')
args = parser.parse_args()

model = LlamaForCausalLM.from_pretrained(args.model)
tokenizer = LlamaTokenizer.from_pretrained(args.model)

# template only relevant for chat models. non-chat models do not need this
TEMPLATE = "You are a helpful assistant. USER: {} ASSISTANT:"
question = "What is the capital of France?"
prompt = TEMPLATE.format(question)
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=300)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

print(output)