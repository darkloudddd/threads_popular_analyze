class Parser:
    def __init__(self):
        self.popular_words = {}

    def parse_words(self, words):
        for word in words:
            if word in self.popular_words:
                self.popular_words[word] += 1
            else:
                self.popular_words[word] = 1
        return self.popular_words

    def get_top_words(self, n=10):
        sorted_words = sorted(self.popular_words.items(), key=lambda item: item[1], reverse=True)
        return sorted_words[:n]