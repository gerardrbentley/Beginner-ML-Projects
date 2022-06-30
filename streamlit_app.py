import streamlit as st

st.header("Beginner Machine Learning (ML) 🤖🧠")

st.subheader(
    "5 ML / AI Topics Demonstrateed in 5 Streamlit Apps."
)
st.text("Learn Python and programming by working on awesome AI applications!")
st.write(f"Built with ❤️ by [Gar's Bar](https://tech.gerardbentley.com/), powered (mainly) by [Streamlit](https://streamlit.io/)!")

st.subheader("1: (NLP) Natural Language Processing 🗣", "nlp")

st.write(
    """'Sentiment Analysis' of any chunk of text.
The 'VADER' model is attuned to sentiments expressed in social media as described in the [source code](https://github.com/cjhutto/vaderSentiment#vader-sentiment-analysis)

Click "Natural Language Processing" on the Sidebar or Run it with `streamlit run pages/01_Natural_Language_Processing.py`

Other awesome app features:
- User text input
- Different behaviour for Positive and Negative

VADER stands for "Valence Aware Dictionary and sEntiment Reasoner"
"""
)

st.subheader("2: (CV) Computer Vision 👀", "cv")

st.write(
    """'Object Detection' on any image.
The 'YOLOv4-tiny' model was trained to locate 80 different types of things (see `yolo/coco.names` for the full list).

Click "Computer Vision" on the Sidebar or Run it with `streamlit run pages/02_Computer_Vision.py`

Other awesome app features:
- Drag and Drop file upload
- Webcam Image processing

YOLO stands for "You Only Look Once"
"""
)

st.subheader("3: (RL) Reinforcement Learning 🤖", "rl")

st.write(
    """'Q-Learning' policy optimization by simulating rounds of BlackJack.
The 'Q-Learning' algorithm will learn to make the best 'Action' choice for a given 'State Observation' in a Reinforcement Learning 'Environment' (see [wikipedia](https://en.wikipedia.org/wiki/Q-learning) for more).

Click "Reinforcement Learning" on the Sidebar or Run it with `streamlit run pages/03_Reinforcement_Learning.py`

Other awesome app features:
- Multiple Views
- Query Arguments in URL

Q roughly stands for "Quality".
It's the mathematical function we aim to learn: the function that predicts a precise 'Reward' based on a combination of 'State' and 'Action'
"""
)

st.subheader("4: Decision Trees 🌲", "trees")

st.write(
    """Use (GBDT) Gradient Boosted Decision Trees to 'classify' a set of 'feature' inputs with a certain output 'label'.
The 'HistGradientBoostingClassifier' model attempts to learn how input features relate to output labels with randomized Decision Trees.
The implementation in [scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) utilizes elements from LightGBM and XGBoost.

Other awesome app features:
- Downloading Models
- Multi-field input forms
"""
)

st.subheader("5: Time Series ⏰", "time-series")

st.write(
    """'Forecasting' future measurements from previous measurements.
The 'Exponential Smoothing' model will learn to fit and predict data in a time series by emphasizing more recent data points during training (see [wikipedia](https://en.wikipedia.org/wiki/Exponential_smoothing) for more).

Other awesome app features:

- Caching computations for faster performance
- Process user data with `pandas` 🐼

This specific model is Holt Winter's Exponential Smoothing, as opposed to Single or Double Exponential Smoothing.
"""
)
