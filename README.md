GitCodeGen: AI-Powered Code Generation Model Using LLMs
Overview
GitCodeGen is an AI-powered code generation model that utilizes Large Language Models (LLMs) to generate high-quality, structured, and syntactically correct code. This project extracts real-world code snippets from GitHub repositories, preprocesses them, and fine-tunes a pre-trained model to improve its ability to generate functional code snippets.

Features
✅ Automated Code Extraction – Scrapes GitHub repositories for Python code snippets using the GitHub API.
✅ LLM Fine-Tuning – Customizes a pre-trained language model (Salesforce CodeGen, GPT-4, etc.) for enhanced code generation.
✅ Context-Aware Code Completion – Generates intelligent, functional, and structured code.
✅ Multi-Language Support – Can be extended to support various programming languages.
✅ Developer Assistance – Automates repetitive coding tasks, improving developer productivity.

Installation
To get started, install the necessary dependencies:


pip install transformers datasets accelerate PyGithub
Usage
1. Collect Code Snippets from GitHub
Use the GitHub API to fetch Python files and extract function definitions:

python

from github import Github
import re
from datasets import Dataset

g = Github("Your_GitHub_Token")
repo = g.get_repo("openai/gym")

def extract_functions_from_code(code):
    pattern = re.compile(r"def\s+(\w+)\s*\(.*\):")
    return pattern.findall(code)

python_files = []
contents = repo.get_contents("")
while contents:
    file_content = contents.pop(0)
    if file_content.type == "dir":
        contents.extend(repo.get_contents(file_content.path))
    elif file_content.path.endswith(".py"):
        python_files.append(file_content)

data = {"code": [], "function_name": []}
for file in python_files:
    code = file.decoded_content.decode("utf-8")
    functions = extract_functions_from_code(code)
    for function in functions:
        data["code"].append(code)
        data["function_name"].append(function)

dataset = Dataset.from_dict(data)
dataset.save_to_disk("code_generation_dataset")
print("Dataset created and saved to disk.")
2. Fine-Tune the Code Generation Model
Load and fine-tune a pre-trained model using the collected dataset:


from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

tokenizer.pad_token = tokenizer.eos_token
dataset = load_from_disk("code_generation_dataset")
dataset = dataset.train_test_split(test_size=0.1)

def preprocess_function(examples):
    return tokenizer(examples['code'], truncation=True, padding='max_length')

tokenized_datasets = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()
3. Generate Code Using the Fine-Tuned Model
Once trained, use the model to generate new code snippets:


def generate_code(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "def fibonacci(n):"
generated_code = generate_code(prompt)

print("Generated Code:")
print(generated_code)
Example Output

Generated Code:
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
Project Structure
bash
Copy
Edit
📂 GitCodeGen
│── 📁 data                # Collected dataset
│── 📁 models              # Trained models
│── 📁 scripts             # Python scripts for training & inference
│── 📄 README.md           # Project documentation
│── 📄 requirements.txt    # Dependencies
│── 📄 train.py            # Model training script
│── 📄 generate.py         # Code generation script
Future Enhancements
🚀 Support for multiple programming languages (Java, C++, JavaScript).
🚀 Deploy as an API for integration with development environments.
🚀 Implement reinforcement learning for better code optimization.

Contributing
Pull requests are welcome! Feel free to open an issue for any suggestions or improvements.

License
This project is licensed under the Apache 2.0 License.

