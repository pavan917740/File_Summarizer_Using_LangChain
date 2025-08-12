# File_Summarizer_Using_LangChain

A Python-based application that generates concise summaries from various file formats using [LangChain](https://github.com/langchain-ai/langchain) and Large Language Models (LLMs). This project is designed to help users quickly extract key information from documents such as PDFs, TXT, DOCX, and more.

## Features

- 📄 **Multi-Format Support:** Summarize PDF, DOCX, TXT, and other common document types.
- 🤖 **Powered by LangChain:** Leverages LangChain's advanced chaining and prompt engineering capabilities for robust document analysis.
- 🧠 **LLM Integration:** Utilizes state-of-the-art LLMs (such as OpenAI GPT models) for high-quality summarization.
- 🛠️ **Customizable:** Easily extend or modify summary prompts and processing logic.
- ⚡ **Batch Processing:** Summarize multiple files in one go.
- 🖥️ **Simple Interface:** Command-line interface for fast and efficient use.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pavan917740/File_Summarizer_Using_LangChain.git
   cd File_Summarizer_Using_LangChain
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your LLM API credentials:**
   - For OpenAI, set the `OPENAI_API_KEY` environment variable.

## Usage

```bash
python summarize.py --input <path_to_file_or_folder> [--output <output_file>] [--model <model_name>]
```

### Examples

- Summarize a single PDF:
  ```bash
  python summarize.py --input example.pdf
  ```

- Summarize all files in a folder and save results to `summaries.txt`:
  ```bash
  python summarize.py --input ./documents --output summaries.txt
  ```

- Specify a custom LLM model:
  ```bash
  python summarize.py --input notes.docx --model gpt-4
  ```

## Configuration

- **Supported file types:** PDF, DOCX, TXT (extendable)
- **LLM Model:** Default is `gpt-3.5-turbo`, configurable via CLI.
- **Prompts and chunking logic:** Can be modified in the source code for custom summarization needs.

## Folder Structure

```
File_Summarizer_Using_LangChain/
├── summarize.py
├── requirements.txt
├── README.md
└── src/
    ├── loaders/
    ├── chains/
    └── utils/
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -am 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request


## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI GPT](https://platform.openai.com/)
- [PyPDF2](https://github.com/py-pdf/PyPDF2) (for PDF parsing)
- [python-docx](https://github.com/python-openxml/python-docx) (for DOCX parsing)
