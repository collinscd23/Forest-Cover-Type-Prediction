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


my_recipe2 <- recipe(Cover_Type~., data = train) %>% 
  step_impute_median(contains("Soil_Type")) %>%
  step_mutate(SoilType = pmax(
    Soil_Type1, Soil_Type2 * 2, Soil_Type3 * 3, Soil_Type4 * 4, Soil_Type5 * 5,
    Soil_Type6 * 6, Soil_Type7 * 7, Soil_Type8 * 8, Soil_Type9 * 9, Soil_Type10 * 10,
    Soil_Type11 * 11, Soil_Type12 * 12, Soil_Type13 * 13, Soil_Type14 * 14, Soil_Type15 * 15,
    Soil_Type16 * 16, Soil_Type17 * 17, Soil_Type18 * 18, Soil_Type19 * 19, Soil_Type20 * 20,
    Soil_Type21 * 21, Soil_Type22 * 22, Soil_Type23 * 23, Soil_Type24 * 24, Soil_Type25 * 25,
    Soil_Type26 * 26, Soil_Type27 * 27, Soil_Type28 * 28, Soil_Type29 * 29, Soil_Type30 * 30,
    Soil_Type31 * 31, Soil_Type32 * 32, Soil_Type33 * 33, Soil_Type34 * 34, Soil_Type35 * 35,
    Soil_Type36 * 36, Soil_Type37 * 37, Soil_Type38 * 38, Soil_Type39 * 39, Soil_Type40 * 40,
    na.rm = TRUE)) %>% 
  step_mutate(WildernessArea = pmax(Wilderness_Area1, Wilderness_Area2 * 2, Wilderness_Area3 * 3, Wilderness_Area4 * 4, na.rm = TRUE)) %>%
  step_rm(contains("Soil_Type")) %>% 
  step_rm(contains("Wilderness_Area")) %>% 
  step_rm(Id) %>% 
  step_mutate(Total_Distance_To_Hydrology = sqrt(Horizontal_Distance_To_Hydrology**2 + Vertical_Distance_To_Hydrology**2)) %>% 
  step_mutate(Elevation_Vertical_Hydrology = Vertical_Distance_To_Hydrology * Elevation) %>% 
  step_mutate(Hydrology_Fire = Horizontal_Distance_To_Hydrology * Horizontal_Distance_To_Fire_Points) %>% 
  step_mutate(Hydrology_Roadways = Horizontal_Distance_To_Hydrology * Horizontal_Distance_To_Roadways) %>% 
  step_mutate(Roadways_Fire = Horizontal_Distance_To_Roadways * Horizontal_Distance_To_Fire_Points) %>% 
  step_mutate_at(c(SoilType, WildernessArea), fn = factor) %>% 
  step_zv(all_predictors())


bake_prep <- prep(my_recipe2)
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
  add_recipe(my_recipe2) 

cv_splits <- vfold_cv(train, v = 3) 

rf_grid <- grid_regular(
  mtry(range = c(2, 15)), 
  min_n(), 
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

vroom_write(submission, "./RandomForestWedn.csv", delim = ",")

