# Web Spider Project

## Overview
This project is a web spider that performs a manual login, scrolls through threads to collect popular words, and outputs the collected information to a file named `result.txt`. The spider is designed to be modular, with separate components handling login, crawling, parsing, and storage of data.

## Project Structure
```
web-spider
├── src
│   ├── main.py          # Entry point of the application
│   ├── login.py         # Handles user login
│   ├── crawler.py       # Manages the crawling process
│   ├── parser.py        # Analyzes collected data
│   ├── storage.py       # Saves data to files
│   └── config
│       └── config.yaml  # Configuration settings
├── tests
│   ├── test_login.py    # Unit tests for login functionality
│   └── test_parser.py   # Unit tests for parser functionality
├── .env.example          # Template for environment variables
├── .gitignore            # Files to ignore in version control
├── requirements.txt      # Project dependencies
├── pyproject.toml       # Project metadata and build system requirements
└── README.md             # Project documentation
```

## Setup Instructions
1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd web-spider
   ```

2. **Install dependencies**:
   Make sure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Configure the project**:
   Update the `src/config/config.yaml` file with the necessary URLs and login credentials.

4. **Run the spider**:
   Execute the main script to start the web spider:
   ```
   python src/main.py
   ```

## Usage Guidelines
- The spider will prompt for manual login credentials.
- Once logged in, it will begin crawling through threads to collect popular words.
- The results will be saved in `result.txt` in the project root directory.

## Analysis
From the repo root (parent of `web-spider`), run:
```
python analyze.py --input web-spider/result.txt --precision
```
This generates `word_tfidf.csv`, `phrase_freq.csv`, `hashtag_freq.csv`, and `word_tfidf.txt`.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
