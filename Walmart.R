library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(lubridate)
library(prophet)

# ============================================================
# Load data
# ============================================================

test <- read_csv("test.csv", show_col_types = FALSE)
train <- read_csv("train.csv", show_col_types = FALSE)
features <- read_csv("features.csv", show_col_types = FALSE)


# ============================================================
# Pre-processing
# ============================================================

features <- features %>%
  mutate(across(starts_with("MarkDown"), ~ replace_na(., 0))) %>%
  mutate(across(starts_with("MarkDown"), ~ pmax(., 0))) %>%
  mutate(
    MarkDown_Total = rowSums(across(starts_with("MarkDown")), na.rm = TRUE),
    MarkDown_Flag  = if_else(MarkDown_Total > 0, 1, 0),
    MarkDown_Log   = log1p(MarkDown_Total)
  ) %>%
  select(-MarkDown1, -MarkDown2, -MarkDown3, -MarkDown4, -MarkDown5)


# Impute CPI and Unemployment
feature_recipe <- recipe(~ ., data = features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment, impute_with = imp_vars(DecDate, Store))

imputed_features <- prep(feature_recipe) %>% juice()%>%
  select(-IsHoliday)


# Join into train/test
train <- train %>%
  left_join(imputed_features, by = c("Store", "Date"))

test <- test %>%
  left_join(imputed_features, by = c("Store", "Date"))


# ============================================================
# LOG TRANSFORM RECIPE FOR MODELING
# ============================================================

store <- 14
dept  <- 7

train_sd <- train |> filter(Store == store, Dept == dept)
test_sd  <- test  |> filter(Store == store, Dept == dept)


sales_recipe <- recipe(Weekly_Sales ~ ., data = train_sd) %>%
  step_mutate(
    Date  = as.numeric(Date),
    IsHoliday  = as.numeric(IsHoliday)
  ) %>%
  step_log(Weekly_Sales, offset = 1)


# ============================================================
# Penalized regression model
# ============================================================

tuned_preg_model <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

tuned_preg_wf <- workflow() %>%
  add_model(tuned_preg_model) %>%
  add_recipe(sales_recipe)


# ============================================================
# CV Setup
# ============================================================

folds <- vfold_cv(train_sd, v = 5)


# ============================================================
# Run tuning
# ============================================================

tuning_grid <- grid_regular(
  penalty(),
  mixture(),
  levels = 10
)

cv_results <- tune_grid(
  tuned_preg_wf,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(rmse)
)


# ============================================================
# REPORT CV RMSE
# ============================================================

cv_summary <- cv_results %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  arrange(mean)

print(cv_summary)

best_cv <- show_best(cv_results, metric = "rmse", n = 1)
print(best_cv)

# best rmse for store 1 dept 1 penreg is 0.293
# for store 7 dept 7 it's 0.405
# for store 14 dept 8 it's 0.0917

# so some combos it does really well, others not so much, but maybe it would average out
# let's try a different model


# ============================================================
# Random forest model
# ============================================================

# Define model with tuning parameters
rf_model <- rand_forest(
  mtry = tune(),       # number of predictors sampled at each split
  trees = 1000,        # number of trees
  min_n = tune()       # min number of samples to split a node
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Workflow
rf_wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(sales_recipe)

# ============================================================
# CV Setup
# ============================================================

folds <- vfold_cv(train_sd, v = 5)

# ============================================================
# Tuning grid
# ============================================================

rf_grid <- grid_regular(
  mtry(range = c(1, ncol(train_sd) - 1)),
  min_n(range = c(2, 10)),
  levels = 5
)

# ============================================================
# Run tuning
# ============================================================

cv_results_rf <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(rmse)
)

# ============================================================
# REPORT CV RMSE
# ============================================================

cv_summary_rf <- cv_results_rf %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  arrange(mean)

print(cv_summary_rf)

best_cv_rf <- show_best(cv_results_rf, metric = "rmse", n = 1)
print(best_cv_rf)

# best rmse for store 1 dept 1 random forest is 0.264
# for store 7 dept 7 it's 0.297
# for store 14 dept 8 it's 0.0841

# this did better than the penalized linear regression for these store/dept combos
# let's try one more model


library(tidymodels)
library(lightgbm)
library(tune)
library(bonsai)
# ============================================================
# LightGBM model
# ============================================================

lgb_model <- boost_tree(
  trees = 100,            # number of trees
  tree_depth = 5,         # max depth
  learn_rate = 0.1,       # learning rate
  mtry = tune(),          # features sampled at each split
  min_n = tune(),         # min samples per node
  loss_reduction = tune() # min gain to split
) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

# Workflow
lgb_wf <- workflow() %>%
  add_model(lgb_model) %>%
  add_recipe(sales_recipe)

# ============================================================
# CV Setup
# ============================================================

folds <- vfold_cv(train_sd, v = 5)

# ============================================================
# Tuning grid
# ============================================================

lgb_grid <- grid_regular(
  mtry(range = c(2, 10)),
  min_n(range = c(2, 20)),
  loss_reduction(range = c(0, 5)),
  levels = 3
)

# ============================================================
# Run tuning
# ============================================================

lgb_cv_results <- tune_grid(
  lgb_wf,
  resamples = folds,
  grid = lgb_grid,
  metrics = metric_set(rmse)
)

# ============================================================
# Report CV RMSE
# ============================================================

cv_summary_lgb <- lgb_cv_results %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  arrange(mean)

print(cv_summary_lgb)

best_cv_lgb <- show_best(lgb_cv_results, metric = "rmse", n = 1)
print(best_cv_lgb)

# best rmse for store 1 dept 1 lightGBM is 0.290
# for store 7 dept 7 it's 0.405
# for store 14 dept 8 it's 0.105

# so the best model so far is random forest



# TRYING THE PROPHET MODEL

library(prophet)

store <- 1
dept <- 3

store2 <- 8
dept2 <- 2

# filter and rename for prophet

sd_train <- train %>%
  filter(Store==store, Dept==dept)%>%
  rename(y=Weekly_Sales, ds=Date)

sd_test <- test %>%
  filter(Store==store, Dept==dept)%>%
  rename(ds=Date)
# filter and rename for prophet

sd_train2 <- train %>%
  filter(Store==store2, Dept==dept2)%>%
  rename(y=Weekly_Sales, ds=Date)

sd_test2 <- test %>%
  filter(Store==store2, Dept==dept2)%>%
  rename(ds=Date)


# fit prophet model

prophet_model <- prophet()%>%
  add_regressor("IsHoliday")%>%
  add_regressor("Fuel_Price")%>%
  add_regressor("MarkDown_Flag")%>%
  fit.prophet(df=sd_train)

# predict using prophet model

fitted_vals <- predict(prophet_model, df=sd_train)
test_preds <- predict(prophet_model, df=sd_test)


# fit prophet model again

prophet_model2 <- prophet()%>%
  add_regressor("IsHoliday")%>%
  add_regressor("Fuel_Price")%>%
  add_regressor("MarkDown_Flag")%>%
  fit.prophet(df=sd_train2)

# predict using prophet model again

fitted_vals2 <- predict(prophet_model2, df=sd_train2)
test_preds2 <- predict(prophet_model2, df=sd_test2)

# make plots

library(patchwork)

p1 <- ggplot() +
  geom_line(data = sd_train, aes(x = ds, y = y, color = "Data")) +
  geom_line(data = fitted_vals, aes(x = as.Date(ds), y = yhat, color = "Fitted")) +
  geom_line(data = test_preds, aes(x = as.Date(ds), y = yhat, color = "Forecast")) +
  scale_color_manual(values = c(
    "Data" = "black",
    "Fitted" = "cornflowerblue",
    "Forecast" = "dark red"
  )) +
  labs(
    title = "Projections for Store 1 Department 3",
    x = "Date",
    y = "Weekly Sales",
    color = ""
  )

p2 <- ggplot() +
  geom_line(data = sd_train2, aes(x = ds, y = y, color = "Data")) +
  geom_line(data = fitted_vals2, aes(x = as.Date(ds), y = yhat, color = "Fitted")) +
  geom_line(data = test_preds2, aes(x = as.Date(ds), y = yhat, color = "Forecast")) +
  scale_color_manual(values = c(
    "Data" = "black",
    "Fitted" = "cornflowerblue",
    "Forecast" = "dark red"
  )) +
  labs(
    title = "Projections for Store 8 Department 2",
    x = "Date",
    y = "Weekly Sales",
    color = ""
  )

p1 + p2

combined_plot <- p1 + p2

ggsave(
  filename = "prophet_forecasts.jpg",
  plot = combined_plot,
  width = 12,        # adjust as needed
  height = 6,        # adjust as needed
  units = "in",
  dpi = 300, 
  bg = "white"
)


# modify dr heatons code

## Libraries
library(tidyverse)
library(vroom)
library(tidymodels)
library(DataExplorer)
library(zoo)
library(lubridate)

## Read in Data
train <- vroom("./train.csv")
test  <- vroom("./test.csv")
features <- vroom("./features.csv")

### Impute Missing Markdowns
features <- features %>%
  mutate(across(starts_with("MarkDown"), ~ replace_na(., 0))) %>%
  mutate(across(starts_with("MarkDown"), ~ pmax(., 0))) %>%
  mutate(
    MarkDown_Total = rowSums(across(starts_with("MarkDown")), na.rm = TRUE),
    MarkDown_Flag  = if_else(MarkDown_Total > 0, 1, 0),
    MarkDown_Log   = log1p(MarkDown_Total)
  ) %>%
  select(-MarkDown1, -MarkDown2, -MarkDown3, -MarkDown4, -MarkDown5)

## Impute Missing CPI & Unemployment
feature_recipe <- recipe(~., data = features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment, impute_with = imp_vars(DecDate, Store))

imputed_features <- juice(prep(feature_recipe))

########################
## Merge into ONE PANEL
########################

# Add placeholder Weekly_Sales to test
test2 <- test %>% mutate(Weekly_Sales = NA_real_)

# Combine train + test
full <- bind_rows(train, test2) %>%
  left_join(imputed_features, by = c("Store", "Date")) %>%
  select(-IsHoliday.y) %>%
  rename(IsHoliday = IsHoliday.x) %>%
  select(-MarkDown_Total) %>%
  arrange(Store, Dept, Date)

##########################
## Create lag/rolling features ONCE globally
##########################

full <- full %>%
  group_by(Store, Dept) %>%
  arrange(Date, .by_group = TRUE) %>%
  mutate(
    # Short lags
    lag_1   = lag(Weekly_Sales, 1),
    lag_2   = lag(Weekly_Sales, 2),
    lag_3   = lag(Weekly_Sales, 3),
    
    # Weekly/seasonal lags
    lag_7   = lag(Weekly_Sales, 7),
    lag_14  = lag(Weekly_Sales, 14),
    lag_21  = lag(Weekly_Sales, 21),
    lag_28  = lag(Weekly_Sales, 28),
    
    # Yearly + multi-year seasonality
    lag_52  = lag(Weekly_Sales, 52),
    lag_104 = lag(Weekly_Sales, 104),
    
    # Rolling features
    roll_3 = rollmeanr(lag_1, 3, fill = NA),
    roll_5 = rollmeanr(lag_1, 5, fill = NA),
    roll_3_sd = rollapplyr(lag_1, 3, sd, fill = NA)
  ) %>%
  ungroup()

########################
## Precompute store/dept averages
########################

store_avg <- full %>%
  filter(!is.na(Weekly_Sales)) %>%
  group_by(Store) %>%
  summarize(store_mean = mean(Weekly_Sales), .groups = "drop")

dept_avg <- full %>%
  filter(!is.na(Weekly_Sales)) %>%
  group_by(Dept) %>%
  summarize(dept_mean = mean(Weekly_Sales), .groups = "drop")

########################
## Split back into Train/Test
########################

fullTrain <- full %>% filter(!is.na(Weekly_Sales))
fullTest  <- full %>% filter(is.na(Weekly_Sales)) %>%
  select(-Weekly_Sales)

##########################
## LOOP THROUGH STORE–DEPT COMBOS
##########################

all_preds <- tibble(Id = character(), Weekly_Sales = numeric())
n_storeDepts <- fullTest %>% distinct(Store, Dept) %>% nrow()
cntr <- 0

for(store in unique(fullTest$Store)){
  
  store_train <- fullTrain %>% filter(Store == store)
  store_test  <- fullTest  %>% filter(Store == store)
  
  for(dept in unique(store_test$Dept)){
    
    dept_train <- store_train %>% filter(Dept == dept)
    dept_test  <- store_test  %>% filter(Dept == dept)
    
    # Add precomputed averages
    dept_train <- dept_train %>%
      left_join(store_avg, by = "Store") %>%
      left_join(dept_avg,  by = "Dept")
    
    dept_test <- dept_test %>%
      left_join(store_avg, by = "Store") %>%
      left_join(dept_avg,  by = "Dept")
    
    # Case 1: No training rows
    if(nrow(dept_train) == 0){
      preds <- dept_test %>%
        transmute(
          Id = paste(Store, Dept, Date, sep = "_"),
          Weekly_Sales = 0
        )
      
      # Case 2: Very small sample
    } else if(nrow(dept_train) < 10){
      preds <- dept_test %>%
        transmute(
          Id = paste(Store, Dept, Date, sep = "_"),
          Weekly_Sales = mean(dept_train$Weekly_Sales)
        )
      
      # Case 3: Random Forest (FAST)
    } else {
      
      my_recipe <- recipe(Weekly_Sales ~ ., data = dept_train) %>%
        step_mutate(Holiday = as.integer(IsHoliday)) %>%
        step_date(Date, features = c("dow","week","month","quarter","year")) %>%
        step_rm(Date, Store, Dept, IsHoliday) %>%
        step_zv()
      
      rf_model <- rand_forest(
        mtry = 5,
        min_n = 5,
        trees = 100
      ) %>%
        set_engine("ranger") %>%
        set_mode("regression")
      
      final_wf <- workflow() %>%
        add_recipe(my_recipe) %>%
        add_model(rf_model) %>%
        fit(dept_train)
      
      preds <- dept_test %>%
        transmute(
          Id = paste(Store, Dept, Date, sep = "_"),
          Weekly_Sales = predict(final_wf, new_data = .) %>% pull(.pred)
        )
    }
    
    all_preds <- bind_rows(all_preds, preds)
    
    cntr <- cntr + 1
    cat("Store", store,
        "Dept", dept,
        "Completed", round(100 * cntr / n_storeDepts, 1), "%\n")
  }
}

## Save predictions
vroom_write(all_preds, "./Predictions4.csv", delim = ",")
