from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
	page_title="California Home Value Predictor",
	page_icon=":house:",
	layout="wide",
	initial_sidebar_state="expanded",
)


MODEL_PATH = Path(__file__).with_name("rfmodel.pkl")
FEATURE_NAMES = [
	"longitude", "latitude", "housing_median_age", "total_rooms",
	"total_bedrooms", "population", "households", "median_income",
	"ocean_proximity_INLAND", "ocean_proximity_ISLAND",
	"ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN",
	"rooms_per_household", "bedrooms_per_room", "population_per_household",
]

# The notebook scaled X before fitting, but did not persist the scaler.
# These are the population statistics from its California housing dataset.
FEATURE_MEANS = np.array([
	-119.569704457364, 35.631861434109, 28.639486434109, 2635.763081395349,
	536.838856589147, 1425.476744186046, 499.539680232558, 3.870671002907,
	0.317393410853, 0.000242248062, 0.110949612403, 0.128779069767,
	5.42899974219, 0.213794099526, 3.070655159436,
])
FEATURE_SCALES = np.array([
	2.003483187747, 2.135900653797, 12.585252725725, 2181.562401735972,
	419.381718111662, 1132.434687757615, 382.320490855257, 1.899775694575,
	0.4654619572, 0.015562434832, 0.314069730969, 0.334955252172,
	2.474113202334, 0.065246586711, 10.385797959128,
])


@st.cache_resource
def load_model():
	with MODEL_PATH.open("rb") as model_file:
		return pickle.load(model_file)


def make_features(values: dict[str, float | str]) -> pd.DataFrame:
	households = max(float(values["households"]), 1.0)
	total_rooms = max(float(values["total_rooms"]), 1.0)
	proximity = values["ocean_proximity"]
	row = {
		"longitude": float(values["longitude"]),
		"latitude": float(values["latitude"]),
		"housing_median_age": float(values["housing_median_age"]),
		"total_rooms": total_rooms,
		"total_bedrooms": float(values["total_bedrooms"]),
		"population": float(values["population"]),
		"households": households,
		"median_income": float(values["median_income"]),
		"ocean_proximity_INLAND": float(proximity == "INLAND"),
		"ocean_proximity_ISLAND": float(proximity == "ISLAND"),
		"ocean_proximity_NEAR BAY": float(proximity == "NEAR BAY"),
		"ocean_proximity_NEAR OCEAN": float(proximity == "NEAR OCEAN"),
		"rooms_per_household": total_rooms / households,
		"bedrooms_per_room": float(values["total_bedrooms"]) / total_rooms,
		"population_per_household": float(values["population"]) / households,
	}
	raw = pd.DataFrame([row], columns=FEATURE_NAMES)
	return (raw - FEATURE_MEANS) / FEATURE_SCALES


def money(value: float) -> str:
	return f"${value:,.0f}"


def collect_inputs() -> dict[str, float | str]:
	st.sidebar.header("Property details")
	st.sidebar.caption("Adjust the values to explore how the forest responds.")
	longitude = st.sidebar.slider("Longitude", -124.35, -114.13, -118.25, 0.01)
	latitude = st.sidebar.slider("Latitude", 32.54, 41.95, 34.05, 0.01)
	age = st.sidebar.slider("Median house age", 1, 52, 28)
	median_income = st.sidebar.slider("Median income (ten-thousands)", 0.5, 15.0, 3.9, 0.1)
	proximity = st.sidebar.selectbox(
		"Ocean proximity", ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
	)
	st.sidebar.subheader("Household composition")
	total_rooms = st.sidebar.number_input("Total rooms", 1, 40000, 2600, 100)
	total_bedrooms = st.sidebar.number_input("Total bedrooms", 1, 10000, 540, 10)
	population = st.sidebar.number_input("Population", 1, 40000, 1400, 50)
	households = st.sidebar.number_input("Households", 1, 10000, 500, 10)
	return {
		"longitude": longitude, "latitude": latitude, "housing_median_age": age,
		"total_rooms": total_rooms, "total_bedrooms": total_bedrooms,
		"population": population, "households": households,
		"median_income": median_income, "ocean_proximity": proximity,
	}


def predict(model, features: pd.DataFrame) -> tuple[float, np.ndarray]:
	predictions = np.array([tree.predict(features)[0] for tree in model.estimators_])
	return float(predictions.mean()), predictions


try:
	model = load_model()
except Exception as error:
	st.error(f"Could not load the saved model: {error}")
	st.stop()

st.title("California Home Value Predictor")
st.write("Explore a Random Forest estimate using the same feature engineering and scaling as the training notebook.")

values = collect_inputs()
features = make_features(values)
prediction, tree_predictions = predict(model, features)
spread = float(tree_predictions.std())

if values["total_bedrooms"] > values["total_rooms"]:
	st.sidebar.warning("Bedrooms exceed rooms. The model will still calculate, but this is unusual.")

st.subheader("Estimated median value")
metric_col, range_col, model_col = st.columns(3)
metric_col.metric("Prediction", money(prediction))
range_col.metric("Forest spread", money(spread), help="Standard deviation across the individual trees.")
model_col.metric("Trees evaluated", f"{len(tree_predictions):,}")

tab_prediction, tab_details, tab_model = st.tabs(["Prediction", "Derived details", "Model signals"])

with tab_prediction:
	st.info(
		f"For a {values['ocean_proximity'].lower()} property near "
		f"{values['latitude']:.2f}, {abs(values['longitude']):.2f}W, "
		f"the estimated median value is **{money(prediction)}**."
	)
	lower = max(0.0, prediction - 2 * spread)
	upper = prediction + 2 * spread
	st.caption(f"A rough model spread is {money(lower)} to {money(upper)} (prediction +/- 2 forest standard deviations).")
	chart_data = pd.DataFrame({"Tree prediction": tree_predictions})
	st.bar_chart(chart_data, y="Tree prediction", height=260)

with tab_details:
	details = pd.DataFrame(
		{
			"Derived feature": ["Rooms per household", "Bedrooms per room", "Population per household"],
			"Value": [
				features["rooms_per_household"].iloc[0] * FEATURE_SCALES[12] + FEATURE_MEANS[12],
				features["bedrooms_per_room"].iloc[0] * FEATURE_SCALES[13] + FEATURE_MEANS[13],
				features["population_per_household"].iloc[0] * FEATURE_SCALES[14] + FEATURE_MEANS[14],
			],
		}
	)
	st.dataframe(details.style.format({"Value": "{:.3f}"}), hide_index=True, use_container_width=True)
	st.caption("The model receives standardized numeric features and one-hot encoded ocean proximity.")

with tab_model:
	importance = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": model.feature_importances_})
	importance = importance.sort_values("Importance", ascending=True)
	st.bar_chart(importance, x="Feature", y="Importance", horizontal=True, height=480)
	st.caption("Feature importance is global to the saved forest, not a guarantee that one input changes the prediction by this amount.")
