class Storage:
    def __init__(self, filename='result.txt'):
        self.filename = filename

    def save_to_file(self, data):
        with open(self.filename, 'w', encoding='utf-8') as file:
            for word in data:
                file.write(f"{word}\n")

    def save_results(self, path, texts, append=False):
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\n", " ") + "\n")
