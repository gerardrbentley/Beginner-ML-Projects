import streamlit as st

st.set_page_config(
    page_title="NLP - Beginner Machine Learning",
    page_icon="🤖",
)
st.header("(NLP) Natural Language Processing Demo 🗣", "nlp")

st.write(
    """'Sentiment Analysis' of any chunk of text.
The 'VADER' model is attuned to sentiments expressed in social media as described in the [source code](https://github.com/cjhutto/vaderSentiment#vader-sentiment-analysis)

Other awesome app features:
- User text input
- Different behaviour for Positive and Negative

VADER stands for "Valence Aware Dictionary and sEntiment Reasoner"

Powered by [NLTK](https://www.nltk.org/index.html) and [Streamlit](https://docs.streamlit.io/).
Built with ❤️ by [Gar's Bar](https://tech.gerardbentley.com/)
"""
)

# First we import Natural Language Toolkit, which gives us easy access to using the VADER analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER Lexicon Data
# see [nltk docs](https://www.nltk.org/data.html) for more information
nltk.download("vader_lexicon")

# Prepare the Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Get User Input
text_input = st.text_area(
    "Enter text to analyze:", "Wow, NLTK + Streamlit is really powerful!"
)
result = analyzer.polarity_scores(text_input)
st.subheader("VADER compound score and raw word scores")
st.write(result)

st.subheader("Display Positive or Negative")
compound_score = result["compound"]
if compound_score > 0.05:
    st.success("That's (probably) a Positive Message! 😃")
elif compound_score < -0.05:
    st.error("That's (probably) a Negative Message! 👿")
else:
    st.warning("That's (probably) a Neutral Message...")

st.write(
    """## Take it further:

- Let users upload a text file or csv of phrases to score
- Retrieve data from a URL or API (ex. Tweet or Youtube Comment) and analyze that
- Dive deeper into NLP tokenization, stop word cleaning, semantic analysis, etc.
- Explore NLP topics such as text generation
"""
)

if st.checkbox("Show Code (~50 lines)"):
    with open(__file__, "r") as f:
        st.code(f.read())
