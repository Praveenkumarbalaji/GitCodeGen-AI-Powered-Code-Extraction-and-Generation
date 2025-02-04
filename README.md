
<body>
  <h1>GitCodeGen: AI-Powered Code Extraction and Generation</h1>
  
  <h2>Overview</h2>
  <p>
    This notebook demonstrates an AI-powered workflow that extracts Python functions from code files in a GitHub repository and generates new code using a fine-tuned language model. The notebook performs the following tasks:
  </p>
  <ul>
    <li>Installs the required packages, including Transformers, Datasets, Accelerate, and PyGithub.</li>
    <li>Fetches Python files from a GitHub repository (e.g., <code>openai/gym</code>) and extracts function definitions using regular expressions.</li>
    <li>Creates a dataset from the extracted code and saves it to disk.</li>
    <li>Loads a pre-trained model (<code>Salesforce/codegen-350M-mono</code>) and its tokenizer for code generation.</li>
    <li>Defines a function to generate code completions given a code prompt.</li>
    <li>Tests code generation with sample prompts such as <code>def merge_sort(arr):</code> and <code>def factorial(n):</code>.</li>
  </ul>

  <h2>Installation &amp; Dependencies</h2>
  <p>
    To run this notebook, ensure you have the following dependencies installed:
  </p>
  <ul>
    <li><code>transformers</code></li>
    <li><code>datasets</code></li>
    <li><code>accelerate</code></li>
    <li><code>PyGithub</code></li>
    <li>Python standard libraries: <code>re</code>, etc.</li>
  </ul>
  <p>
    You can install these packages using pip. For example:
  </p>
  <pre><code>!pip install transformers datasets accelerate
!pip install PyGithub</code></pre>

  <h2>Usage</h2>
  <ol>
    <li>
      <strong>Open in Colab:</strong> Click the Colab badge in the first cell of the notebook to open it in Google Colab:
      <br>
      <a href="https://colab.research.google.com/github/Praveenkumarbalaji/GitCodeGen-AI-Powered-Code-Extraction-and-Generation/blob/main/GitCodeGen_AI_Powered_Code_Extraction_and_Generation.ipynb" target="_blank">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
      </a>
    </li>
    <li>
      <strong>Run All Cells:</strong> Execute the notebook cells sequentially. The notebook will:
      <ul>
        <li>Install and import necessary packages.</li>
        <li>Extract functions from Python files fetched from GitHub.</li>
        <li>Create a code generation dataset and save it to disk.</li>
        <li>Load the pre-trained model and tokenizer from Hugging Face.</li>
        <li>Generate code completions for provided prompts.</li>
      </ul>
    </li>
    <li>
      <strong>Test Code Generation:</strong> Modify the prompt strings as needed and observe the generated code output.
    </li>
  </ol>

  <h2>Notebook Structure</h2>
  <p>
    The notebook is organized into several sections:
  </p>
  <ul>
    <li><strong>Colab Badge:</strong> A clickable badge that opens the notebook in Google Colab.</li>
    <li><strong>Dependency Installation:</strong> Code cells to install required packages and libraries.</li>
    <li><strong>GitHub Code Extraction:</strong> A section that uses PyGithub to fetch Python files from a repository and extracts function definitions using regular expressions.</li>
    <li><strong>Dataset Creation:</strong> Code to build a code generation dataset from the extracted functions and save it locally.</li>
    <li><strong>Model Setup &amp; Code Generation:</strong> Loading the <code>Salesforce/codegen-350M-mono</code> model and tokenizer, defining a code generation function, and testing the model with sample prompts.</li>
  </ul>

  <h2>Acknowledgements</h2>
  <p>
    This project utilizes open-source libraries including Transformers, Datasets, Accelerate, and PyGithub. Thanks to the respective communities for making these tools available.
  </p>

  <h2>License</h2>
  <p>
    This project is licensed under the MIT License.
  </p>
</body>
</html>
