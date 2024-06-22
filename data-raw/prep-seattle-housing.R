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
  filter(date >= ymd("2015-01-01")) |> 
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

library(vetiver)

v <- vetiver_model(housing_fit, "julia.silge/seattle-housing-rstats")
board <- board_connect()
board |> vetiver_pin_write(v)

vetiver_deploy_rsconnect(
  board = board, 
  name = "julia.silge/seattle-housing-rstats",
  predict_args = list(debug = TRUE),
  account = "julia.silge",
  appName = "seattle-housing-rstats-model-api",
  forceUpdate = TRUE
)
