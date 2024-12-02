library(tidyverse)
library(tidymodels)
library(vroom)
library(naivebayes)
library(discrim)
# Set working directory
setwd("~/Desktop/Stats348/Forest-Cover-Type-Prediction")

train <- vroom("train.csv")
test <- vroom("test.csv")

train$Cover_Type <- as.factor(train$Cover_Type)

my_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  step_normalize(
    Elevation, Aspect, Slope,
    Horizontal_Distance_To_Hydrology,
    Vertical_Distance_To_Hydrology,
    Horizontal_Distance_To_Roadways,
    Hillshade_9am, Hillshade_Noon,
    Hillshade_3pm,
    Horizontal_Distance_To_Fire_Points
  ) %>% # Normalize numeric features
  step_zv(all_predictors()) %>%  # Remove zero-variance predictors
  step_nzv(all_predictors()) %>% # Remove near-zero variance predictors
  step_dummy(all_nominal_predictors(), -all_outcomes()) # Convert categorical variables to dummies

nb_model <- naive_Bayes(
  Laplace = tune(),
  smoothness = tune()
) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(
  Laplace(range = c(0.1, 10)),
  smoothness(range = c(0.1, 5)),
  levels = 10
)

folds <- vfold_cv(train, v = 5)

cv_results <- tune_grid(
  nb_workflow,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc, accuracy)
)

best_tune <- select_best(cv_results, metric = "roc_auc")

final_workflow <- finalize_workflow(nb_workflow, best_tune)

final_fit <- fit(final_workflow, data = train)

nb_preds <- predict(final_fit, new_data = test, type = "class")

nb_submission <- test %>%
  select(Id) %>% # Ensure 'Id' is selected from the test dataset
  bind_cols(Cover_Type = nb_preds$.pred_class) # Rename the predicted column to 'Cover_Type'

vroom_write(nb_submission, file = "NB_Submission.csv", delim = ",")
