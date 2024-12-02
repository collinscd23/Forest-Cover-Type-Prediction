library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)

setwd("~/Desktop/Stats348/Forest-Cover-Type-Prediction")

# Load Data
train <- vroom("train.csv")
test <- vroom("test.csv")

train$Cover_Type <- as.factor(train$Cover_Type)

# Recipe
boosted_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  step_zv(all_predictors()) %>%  # Remove zero-variance columns
  step_normalize(all_numeric_predictors(), -all_outcomes()) %>%
  step_pca(starts_with("Soil_Type"), num_comp = 5) %>%  # PCA on Soil_Type columns
  step_smote(Cover_Type) %>%  # Balance classes using SMOTE
  step_mutate_at(starts_with("Wilderness_Area"), fn = as.factor) %>%  # Convert to factors
  step_dummy(starts_with("Wilderness_Area"), one_hot = TRUE)  # One-hot encode Wilderness_Area

boosted_recipe1 <- recipe(Cover_Type ~ ., data = train) %>%
  step_zv(all_predictors()) %>%  # Remove zero-variance columns
  step_normalize(all_numeric_predictors(), -all_outcomes()) %>%  # Normalize numeric predictors
  step_mutate_at(starts_with("Wilderness_Area"), fn = as.factor) %>%  # Convert Wilderness_Area to factors
  step_dummy(starts_with("Wilderness_Area"), one_hot = TRUE)  # One-hot encode Wilderness_Area



# Boosted Tree Specification
boosted_spec <- boost_tree(
  trees = tune(),              # Number of trees
  tree_depth = tune(),         # Depth of trees
  learn_rate = tune(),         # Learning rate
  mtry = tune(),               # Number of predictors to randomly sample
  loss_reduction = tune(),     # Minimum reduction in loss for a split
  sample_size = tune()         # Proportion of training data used in each tree
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Workflow
boosted_workflow <- workflow() %>%
  add_model(boosted_spec) %>%
  add_recipe(boosted_recipe)

# Cross-Validation
cv_splits <- vfold_cv(train, v = 5)

# Grid for Tuning
boosted_grid <- grid_regular(
  trees(range = c(100, 200)),
  tree_depth(range = c(3, 10)),
  learn_rate(range = c(0.01, 0.3)),
  mtry(range = c(2, 10)),
  loss_reduction(range = c(0, 10)),
  sample_size = sample_prop(range = c(0.5, 1)),  # Proper proportion specification
  levels = 5  # Number of levels for each hyperparameter
)


# Hyperparameter Tuning
boosted_results <- tune_grid(
  boosted_workflow,
  resamples = cv_splits,
  grid = boosted_grid,
  metrics = metric_set(accuracy, roc_auc)
)

# Select Best Model
best_boosted <- boosted_results %>%
  select_best(metric = "accuracy")

# Final Workflow
final_boosted_workflow <- boosted_workflow %>%
  finalize_workflow(best_boosted)

# Fit Final Model
final_boosted_model <- final_boosted_workflow %>%
  fit(data = train)

# Predictions for Submission
test_predictions <- predict(final_boosted_model, new_data = test, type = "class")

submission <- test %>%
  select(Id) %>%
  bind_cols(Cover_Type = test_predictions$.pred_class)

vroom_write(submission, "./BoostedTreesSubmission.csv", delim = ",")

