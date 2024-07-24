import os
import pins
import rsconnect
import shap
import vetiver
import pandas as pd
import numpy as np

from sklearn import ensemble, model_selection
from vetiver import VetiverModel, vetiver_pin_write
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CONNECT_API_KEY")
url = os.getenv("CONNECT_SERVER")
connect_server = rsconnect.api.RSConnectServer(url=url, api_key=api_key)
housing = pd.read_parquet("./data/housing.parquet", engine="pyarrow")

X, y = housing[["bedrooms", "bathrooms", "sqft_living", "yr_built"]], np.log10(housing["price"])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

housing_fit = ensemble.RandomForestRegressor(n_estimators=200).fit(X_train, y_train)


v = VetiverModel(
    housing_fit, "isabel.zimmerman/seattle-housing-python", prototype_data=X_train
)
board = pins.board_connect(server_url=url, api_key=api_key, allow_pickle_read=True)

vetiver_pin_write(board, v)

explainer = shap.TreeExplainer(housing_fit, X_train)
big_house = pd.DataFrame({'bedrooms': [4], 'bathrooms': [3.5], 'sqft_living': [3e3], 'yr_built': [1999]})
shap_values = explainer(big_house)
shap.plots.bar(shap_values, show=False)

board.pin_write(explainer,"isabel.zimmerman/seattle-shap-python", type="joblib")

vetiver.deploy_rsconnect(
    connect_server=connect_server,
    board=board,
    pin_name="isabel.zimmerman/seattle-housing-python",
    title="seattle-housing-python-model-api",
    extra_files=["./data-raw/requirements.txt"],
)
