library(tidyverse)
library(arrow)
library(pins)

housing <- tibble::as_tibble(mlr3data::kc_housing) |> 
  mutate(date = as.Date(date)) |> 
  select(price, date, bedrooms, bathrooms, sqft_living, yr_built, waterfront, lat, long)

summary(housing$date)

path1 <- here::here("data", "housing.parquet")
path2 <- here::here("data", "housing_monitoring.parquet")

housing |> 
  filter(date < ymd("2015-01-01")) |>
  arrange(date) |> 
  write_parquet(path1)

housing |> 
  filter(date >= ymd("2015-01-01"), date < "2015-05-14") |> 
  arrange(date) |> 
  write_parquet(path2)

library(tidymodels)
set.seed(123)
housing_split <- housing |> 
  filter(date < ymd("2015-01-01")) |>
  mutate(price = log10(price)) |> 
  arrange(date) |> 
  initial_split(prop = 0.8)

housing_train <- training(housing_split)
housing_test <- testing(housing_split)

housing_fit <-
  workflow(
    price ~ bedrooms + bathrooms + sqft_living + yr_built, 
    rand_forest(trees = 200, mode = "regression")
  ) |> 
  fit(data = housing_train)

augment(housing_fit, new_data = slice_sample(housing_test, n = 10))

library(DALEXtra)

explainer_rf <-
  explain_tidymodels(
    housing_fit,
    data = housing_train |> select(bedrooms, bathrooms, sqft_living, yr_built),
    y = housing_train$price,
    label = "Seattle housing random forest",
    verbose = FALSE
  )

big_house <- tibble(bedrooms = 4, bathrooms = 3.5, sqft_living = 3e3, yr_built = 1999)
shap <- predict_parts(explainer_rf, big_house, type = "shap", B = 25)
plot(shap)

library(vetiver)

v <- vetiver_model(housing_fit, "julia.silge/seattle-housing-rstats")
board <- board_connect()
board |> vetiver_pin_write(v)
board |> pin_write(explainer_rf, "julia.silge/seattle-shap-rstats")

vetiver_deploy_rsconnect(
  board = board, 
  name = "julia.silge/seattle-housing-rstats",
  predict_args = list(debug = TRUE),
  account = "julia.silge",
  appName = "seattle-housing-rstats-model-api",
  forceUpdate = TRUE
)
