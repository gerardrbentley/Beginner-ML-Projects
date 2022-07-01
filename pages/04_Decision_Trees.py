import streamlit as st

st.set_page_config(
    page_title="Decision Trees - Beginner Machine Learning",
    page_icon="🤖",
)
st.header("Decision Trees Demo 🌲", "nlp")

st.write(
    """Use (GBDT) Gradient Boosted Decision Trees to 'classify' a set of 'feature' inputs with a certain output 'label'.
The 'HistGradientBoostingClassifier' model attempts to learn how input features relate to output labels with randomized Decision Trees.
The implementation in [scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) utilizes elements from LightGBM and XGBoost.

This Example:

- Use Penguin 'features' such as Bill Length and Body Mass
- 'label' a set of observed features with the most likely Penguin species
- Train the 'Classifier' model on a set of real observations from [Palmer Station](https://allisonhorst.github.io/palmerpenguins/) in Antarctica!

Other awesome app features:
- Downloading Models
- Multi-field input forms

Powered by [Scikit-Learn](https://scikit-learn.org/stable/index.html) and [Streamlit](https://docs.streamlit.io/).
Built with ❤️ by [Gar's Bar](https://tech.gerardbentley.com/)
"""
)

import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def render_end():
    st.write(
        """## Take it further:

- Compare results to simpler tabular models (Regression, Random Forest) or other Gradient Boosting implementations (XGBoost, LightGBM, CatBoost)
- Perform a 'regression' task instead of 'classification' to get a pseudo-confidence score from the model
- Utilize Cross-Validation with Quantile Losses or Mean losses to assess confidence intervals
- Explore tabular models in other use cases such as Time Series analysis
    """
    )

    if st.checkbox("Show Code (~200 lines)"):
        with open(__file__, "r") as f:
            st.code(f.read())
    st.stop()


@st.experimental_memo
def run_training(features, labels):
    model = HistGradientBoostingClassifier(
        random_state=47, categorical_features=categorical_column_mask
    )
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, random_state=47
    )
    model.fit(features_train, labels_train)
    labels_predicted = model.predict(features_test)
    labels_predicted_train = model.predict(features_train)
    test_score = accuracy_score(labels_predicted, labels_test)
    train_score = accuracy_score(labels_predicted_train, labels_train)
    feature_importance = get_importances(model, features_train, labels_train)

    labels_predicted_display = species_encoder.inverse_transform(labels_predicted)
    labels_test_display = species_encoder.inverse_transform(labels_test)
    results = pd.DataFrame(
        {
            "Predicted Species": labels_predicted_display,
            "Actual Species": labels_test_display,
            **features_test[feature_cols],
        }
    )
    return model, results, feature_importance, train_score, test_score


def get_importances(model, features, labels):
    result = permutation_importance(
        model, features, labels, n_repeats=10, random_state=47
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=features.columns[sorted_importances_idx],
    )
    return importances


penguin_df = pd.read_csv(
    "data/penguins.csv",
    dtype={"species": "category", "island": "category", "sex": "category"},
)
penguin_df = penguin_df.dropna()

with st.expander("Raw Data"):
    penguin_df

if st.checkbox("Show Feature Comparison", True):
    x_label = st.selectbox(
        "X Axis Feature",
        penguin_df.columns,
        penguin_df.columns.get_loc("bill_length_mm"),
    )
    y_label = st.selectbox(
        "Y Axis Feature",
        penguin_df.columns,
        penguin_df.columns.get_loc("bill_depth_mm"),
    )
    fig, ax = plt.subplots()
    penguin_df.plot.scatter(x_label, y_label, c="species", ax=ax, colormap="rainbow")
    ax.set_title(f"Scatter Plot Feature Comparison: {x_label} x {y_label}")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.axhline(y=0, color="k", linestyle="--")
    ax.figure.tight_layout()
    st.pyplot(fig)

species_encoder = LabelEncoder()
island_encoder = LabelEncoder()
sex_encoder = LabelEncoder()

labels = species_encoder.fit_transform(penguin_df["species"])
feature_cols = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
categorical_column_mask = [column in ("island", "sex") for column in feature_cols]

features = penguin_df[feature_cols]
features["island"] = island_encoder.fit_transform(penguin_df["island"])
features["sex"] = sex_encoder.fit_transform(penguin_df["sex"])


if not st.checkbox("Press Here to Train Model on 75% of the data"):
    render_end()

model, results, feature_importance, train_score, test_score = run_training(
    features, labels
)
with st.expander("Missed Predictions"):
    results[results["Predicted Species"] != results["Actual Species"]]

if st.checkbox("Show Feature Importance", key="feat_importance"):
    fig, ax = plt.subplots()
    feature_importance.plot.box(vert=False, whis=10, ax=ax)
    ax.set_title("Permutation Importances")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    st.pyplot(fig)

st.download_button(
    f"Download Trained Model",
    pickle.dumps(model),
    help="Download the model weights to be used for future predictions",
)

st.header("Predict your own Penguin's species 🐧")

with st.form("user_inputs"):
    island = st.selectbox("Penguin Island", options=list(penguin_df["island"].unique()))
    sex = st.selectbox("Sex", options=list(penguin_df["sex"].unique()))
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    is_submitted = st.form_submit_button()

if not is_submitted:
    st.info("Hit 'Submit' to predict your Penguin's species")
else:
    feature_input = {
        "island": island,
        "bill_length": bill_length,
        "bill_depth": bill_depth,
        "flipper_length": flipper_length,
        "body_mass": body_mass,
        "sex": sex,
    }
    st.json(feature_input)
    feature_input["island"] = island_encoder.transform([island])[0]
    feature_input["sex"] = sex_encoder.transform([sex])[0]
    with st.expander("Show Transformed User Input"):
        feature_input
    prediction = model.predict([[*feature_input.values()]])
    with st.expander("Raw Prediction"):
        prediction[0]
    (species,) = species_encoder.inverse_transform([prediction])
    st.success(f"Most likely an **{species}** Penguin!")

render_end()
