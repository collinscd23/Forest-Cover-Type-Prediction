library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)
library(recipes)

setwd("~/Desktop/Stats348/Forest-Cover-Type-Prediction")

train <- vroom("train.csv")
test <- vroom("test.csv")

train$Cover_Type <- as.factor(train$Cover_Type)

my_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  step_zv(all_predictors()) %>%  # Remove zero-variance columns
  step_normalize(all_numeric_predictors(), -all_outcomes(), -starts_with("Wilderness_Area"), -starts_with("Soil_Type")) %>%
  step_mutate(
    Hydro_Euclidean_Dist = sqrt(Horizontal_Distance_To_Hydrology^2 + Vertical_Distance_To_Hydrology^2),
    Hillshade_Interaction = Hillshade_9am * Hillshade_Noon * Hillshade_3pm,
    Aspect_sin = sin(Aspect * pi / 180),
    Aspect_cos = cos(Aspect * pi / 180),
    Terrain_Roughness = Slope * Elevation,
    Relative_Distance_Water_Road = Horizontal_Distance_To_Hydrology - Horizontal_Distance_To_Roadways
  ) %>%
  step_mutate_at(starts_with("Wilderness_Area"), fn = as.factor) %>%
  step_mutate_at(starts_with("Soil_Type"), fn = as.factor) %>%
  step_dummy(starts_with("Wilderness_Area"), starts_with("Soil_Type")) %>%
  step_pca(starts_with("Soil_Type"), num_comp = 5) %>%
  step_smote(Cover_Type)

my_recipe1 <- recipe(Cover_Type ~ ., data = train) %>%
  step_zv(all_predictors()) %>%  # Remove zero-variance columns
  step_normalize(all_numeric_predictors(), -all_outcomes(), -starts_with("Wilderness_Area"), -starts_with("Soil_Type")) %>%
  step_pca(starts_with("Soil_Type"), num_comp = 5) %>%  # Perform PCA on Soil_Type columns
  step_smote(Cover_Type) %>%  # Apply SMOTE while all columns are numeric
  step_mutate_at(starts_with("Wilderness_Area"), fn = as.factor) %>%  # Convert Wilderness_Area columns to factors afterward
  step_mutate_at(starts_with("Soil_Type"), fn = as.factor)

bake_prep <- prep(my_recipe1)
baked <- bake(bake_prep, new_data = NULL)


rf_spec <- rand_forest(
  mtry = tune(), 
  trees = 500,   
  min_n = tune() 
) %>%
  set_engine("ranger") %>% 
  set_mode("classification") 


rf_workflow <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(my_recipe1) 

cv_splits <- vfold_cv(train, v = 2) 

rf_grid <- grid_regular(
  mtry(range = c(2, 10)), 
  min_n(range = c(5, 15)), 
  levels = 5
)


rf_results <- tune_grid(
  rf_workflow,
  resamples = cv_splits,
  grid = rf_grid,
  metrics = metric_set(accuracy)
)

best_rf <- rf_results %>%
  select_best(metric = "accuracy")

final_rf_workflow <- rf_workflow %>%
  finalize_workflow(best_rf)

final_rf_model <- final_rf_workflow %>%
  fit(data = train)

test_predictions <- predict(final_rf_model, new_data = test, type = "class")

submission <- test %>%
  select(Id) %>%                    
  bind_cols(Cover_Type = test_predictions$.pred_class) 

vroom_write(submission, "./RandomForestNEW.csv", delim = ",")

