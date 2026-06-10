import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
domain_stopwords = {"study", "paper", "method", "results", "analysis"}

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)

    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if w not in domain_stopwords]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 2]

    return " ".join(tokens)


# ✅ Quick Test (put at bottom)
if __name__ == "__main__":
    sample = "This study proposes a novel deep learning method for soil classification."
    print(clean_text(sample))
