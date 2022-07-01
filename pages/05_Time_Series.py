import streamlit as st

st.set_page_config(
    page_title="Time Series - Beginner Machine Learning",
    page_icon="🤖",
)
st.header("Time Series Demo ⏰", "time-series")

st.write(
    """'Forecasting' future measurements from previous measurements.
The 'Exponential Smoothing' model will learn to fit and predict data in a time series by emphasizing more recent data points during training (see [wikipedia](https://en.wikipedia.org/wiki/Exponential_smoothing) for more).

Other awesome app features:

- Caching computations for faster performance
- Process user data with `pandas` 🐼

This specific model is Holt Winter's Exponential Smoothing, as opposed to Single or Double Exponential Smoothing.

Powered by [unit8 Darts](https://unit8co.github.io/darts/README.html) + [statsmodel](https://www.statsmodels.org/stable/tsa.html) and [Streamlit](https://docs.streamlit.io/).
Built with ❤️ by [Gar's Bar](https://tech.gerardbentley.com/)
"""
)


import pandas as pd
import plotly.express as px
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode

trends = {
    "Additive": ModelMode.ADDITIVE,
    "Multiplicative": ModelMode.MULTIPLICATIVE,
    "None": ModelMode.NONE,
}

seasonals = {
    "Additive": SeasonalityMode.ADDITIVE,
    "Multiplicative": SeasonalityMode.MULTIPLICATIVE,
    "None": SeasonalityMode.NONE,
}


@st.experimental_memo
def load_csv_data(csv_data, delimiter):
    return pd.read_csv(csv_data, sep=delimiter)


@st.experimental_memo
def get_prediction(
    model_cls, _model_args, _model_kwargs, train, num_predictions, num_samples
):
    model = model_cls(*_model_args, **_model_kwargs)
    model.fit(TimeSeries.from_dataframe(train))
    prediction = model.predict(num_predictions, num_samples=num_samples)
    return prediction


@st.experimental_memo
def get_historical_forecast(
    model_cls,
    _model_args,
    _model_kwargs,
    timeseries,
    start,
    forecast_horizon,
    stride,
    retrain,
):
    model = model_cls(*_model_args, **_model_kwargs)
    historical_forecast = model.historical_forecasts(
        TimeSeries.from_dataframe(timeseries),
        start=start,
        forecast_horizon=forecast_horizon,
        stride=stride,
        retrain=retrain,
        overlap_end=False,
        last_points_only=True,
    )
    historical_df = historical_forecast.pd_dataframe()
    return historical_forecast, historical_df


def render_end():
    st.write(
        """## Take it further:

- Explore the model's performance based on different error metrics
- Perform a grid search over different model parameters to find the best fit
- Create an ensemble model to forecast using predictions from multiple models
- Explore Neural Network models used for forecasting applications"""
    )

    if st.checkbox("Show Code (~340 lines)"):
        with open(__file__, "r") as f:
            st.code(f.read())
    st.stop()


raw_dataset = st.sidebar.file_uploader(
    "Upload dataset CSV file", accept_multiple_files=False
)

delimiter = st.sidebar.text_input(
    "CSV Delimiter", value=",", key="Character delimiter for the CSV data"
)

if raw_dataset is None:
    st.info(
        """\
# Start Here
        
Upload a CSV file in the sidebar to continue.

[Right-Click Here](https://raw.githubusercontent.com/unit8co/darts/master/datasets/us_gasoline.csv) to 'Save Link As' a Weekly U.S. Product Supplied of Finished Motor Gasoline Dataset. 

Then drag it into the drop zone on the left.

Or upload your own CSV with a time column and at least one value column to forecast!"""
    )
    render_end()

df = load_csv_data(raw_dataset, delimiter)
with st.expander("Show Raw Dataset"):
    df
if len(df.columns) == 1:
    st.error(
        "Dataset appears to have 1 column. Need one Time Column and at least one Value Column"
    )
    render_end()
if len(df.columns) == 2:
    st.sidebar.info(
        f"Univariate Dataset, the non-Time Column will be used as the Value Column"
    )
with st.sidebar.form("form"):
    time_col = st.selectbox(
        "Time Column",
        df.columns,
        help="Name of the column in your csv with time step values",
    )
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col, drop=True)
    columns = list(df.columns)

    if len(columns) == 1:
        value_cols = [columns[0]]
    else:
        value_cols = st.multiselect(
            "Values Column(s)",
            columns,
            columns[0],
            key="value_column",
            help="Name of column(s) with values to sample and forecast",
        )
    options = {
        "Weekly": ("W", 52),
        "Monthly": ("M", 12),
        "Yearly": ("A", 1),
        "Daily": ("D", 365),
        "Hourly": ("H", 365 * 24),
        "Quarterly": ("Q", 8),
    }
    sampling_period = st.selectbox(
        "Time Series Period",
        options,
        help="How to define samples. Pandas will average entries between time steps to create a well-formed Time Series",
    )
    freq_string, periods_per_year = options[sampling_period]
    # trend (Optional[ModelMode]) –
    trend_choice = st.radio(
        "Trend Type",
        trends,
        help="Type of trend component. Either ADDITIVE, MULTIPLICATIVE, NONE, or None. Defaults to ADDITIVE.",
    )
    seasonal_choice = st.radio(
        "Seasonality Type",
        seasonals,
        help="Type of seasonal component. Either ADDITIVE, MULTIPLICATIVE, NONE, or None. Defaults to ADDITIVE.",
    )
    # q = st.number_input("Autoregressive Period", value=52, min_value=0, help="Order (number of time lags) of the autoregressive model (AR). 52 is one year in weeks.")

    df = df.resample(freq_string).mean()
    timeseries = TimeSeries.from_dataframe(df, value_cols=value_cols)
    forecast_horizon = int(
        st.number_input(
            "Forecast Horizon",
            key="forecast_horizon",
            value=16,
            min_value=1,
            max_value=len(timeseries),
            help="(For Historical Forecast) How many time steps separate the prediction time from the forecast time (ex. predicting 16 weeks into the future",
        )
    )
    stride = int(
        st.number_input(
            "Historical Forecast Stride",
            key="stride",
            value=1,
            min_value=1,
            max_value=forecast_horizon,
            help="(For Historical Forecast) How many time steps between two consecutive predictions",
        )
    )
    num_predictions = int(
        st.number_input(
            "Number of Time Steps to Forecast",
            key="num_predictions",
            value=104,
            min_value=1,
            max_value=len(timeseries),
            help="How many time steps worth of datapoints to exclude from training",
        )
    )
    num_samples = int(
        st.number_input(
            "Number of Prediction Samples",
            key="cust_sample",
            min_value=1,
            max_value=10000,
            value=100,
            help="Number of times a prediction is sampled for a probabilistic model",
        )
    )
    st.subheader("Customize Plotting")
    low_quantile = st.slider(
        "Lower Percentile",
        key="low_quantile",
        min_value=0.01,
        max_value=0.99,
        value=0.05,
        help="The quantile to use for the lower bound of the plotted confidence interval.",
    )
    mid_quantile = st.slider(
        "Middle Percentile",
        key="mid_quantile",
        min_value=0.01,
        max_value=0.99,
        value=0.5,
        help="The quantile to use for the center of the plotted confidence interval.",
    )
    high_quantile = st.slider(
        "High Percentile",
        key="high_quantile",
        min_value=0.01,
        max_value=0.99,
        value=0.95,
        help="The quantile to use for the upper bound of the plotted confidence interval.",
    )
    st.checkbox(
        f"Show {num_predictions} Forecasted Steps",
        value=True,
        key="show_forecast",
        help=f"Forecasted {num_predictions} from the end of the training set ({timeseries[-num_predictions]}).",
    )
    st.checkbox(
        "Show Historical Forecast",
        value=True,
        key="show_historical",
        help="Starting from the end of the training set, incrementally refit the model on new future values.",
    )
    is_submitted = st.form_submit_button("Launch Forecasting")

if not is_submitted:
    st.info(
        """\
# Next Step

Dataset uploaded!

Validate it looks good with the 'Show Raw Dataset' expander above.

Then choose which column name should be used for time stamps and which column name(s) should be used for values for forecast.

Hit 'Launch Forecasting' to run it!
"""
    )
    render_end()
train, val = timeseries[:-num_predictions], timeseries[-num_predictions:]
st.info("Training Model")

if st.session_state.show_forecast:
    prediction = get_prediction(
        ExponentialSmoothing,
        (trends[trend_choice], seasonals[seasonal_choice]),
        {},
        train.pd_dataframe(),
        num_predictions,
        num_samples,
    )
    st.success("Forecasted Data")
    prediction_df = prediction.quantiles_df([low_quantile, mid_quantile, high_quantile])
if st.session_state.show_historical:
    st.info("Running Historical Forecast")
    historical_forecast, historical_df = get_historical_forecast(
        ExponentialSmoothing,
        (trends[trend_choice], seasonals[seasonal_choice]),
        {},
        timeseries.pd_dataframe(),
        timeseries.n_timesteps - num_predictions,
        forecast_horizon,
        stride,
        True,
    )
display_data = timeseries.pd_dataframe().rename(lambda c: f"observation_{c}", axis=1)
st.subheader("Data and Forecast Plot")


if st.session_state.show_forecast:
    display_data = display_data.join(
        prediction_df.rename(lambda c: f"prediction_{c}", axis=1)
    )


if st.session_state.show_historical:
    display_data = display_data.join(
        historical_df.rename(lambda c: f"historical_forecast_{c}", axis=1)
    )
    display_data.iloc[-num_predictions:] = display_data.iloc[-num_predictions:].fillna(
        method="ffill"
    )
st.text("Double click to zoom in / back out")

title = f"Predicting {sampling_period} {'. '.join(value_cols)} with Exponential Smoothing model"

fig = px.line(
    display_data,
    title=title,
    labels={
        "value": ", ".join(value_cols),
        display_data.index.name: sampling_period[:-2],
    },
)
start_date = display_data.index[-num_predictions * 2]
end_date = display_data.index[-1]

fig.update_xaxes(type="date", range=[start_date, end_date])
st.plotly_chart(fig)
render_end()
