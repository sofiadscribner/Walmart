7## Libraries I need
library(tidyverse)
library(vroom)
library(tidymodels)
library(DataExplorer)

## Read in the Data
train <- vroom("./train.csv")
test <- vroom("./test.csv")
features <- vroom("./features.csv")

#########
## EDA ##
#########
plot_missing(features)
plot_missing(test)

### Impute Missing Markdowns
features <- features %>%
  mutate(across(starts_with("MarkDown"), ~ replace_na(., 0))) %>%
  mutate(across(starts_with("MarkDown"), ~ pmax(., 0))) %>%
  mutate(
    MarkDown_Total = rowSums(across(starts_with("MarkDown")), na.rm = TRUE),
    MarkDown_Flag = if_else(MarkDown_Total > 0, 1, 0),
    MarkDown_Log   = log1p(MarkDown_Total)
  ) %>%
  select(-MarkDown1, -MarkDown2, -MarkDown3, -MarkDown4, -MarkDown5)

## Impute Missing CPI and Unemployment
feature_recipe <- recipe(~., data=features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment,
                  impute_with = imp_vars(DecDate, Store))
imputed_features <- juice(prep(feature_recipe))

########################
## Merge the Datasets ##
########################

fullTrain <- left_join(train, imputed_features, by=c("Store", "Date")) %>%
  select(-IsHoliday.y) %>%
  rename(IsHoliday=IsHoliday.x) %>%
  select(-MarkDown_Total)
fullTest <- left_join(test, imputed_features, by=c("Store", "Date")) %>%
  select(-IsHoliday.y) %>%
  rename(IsHoliday=IsHoliday.x) %>%
  select(-MarkDown_Total)
plot_missing(fullTrain)
plot_missing(fullTest)

##################################
## Loop Through the Store-depts ## 
## and generate predictions.    ##
##################################
all_preds <- tibble(Id = character(), Weekly_Sales = numeric())
n_storeDepts <- fullTest %>% distinct(Store, Dept) %>% nrow()
cntr <- 0
for(store in unique(fullTest$Store)){
  
  store_train <- fullTrain %>%
    filter(Store==store)
  store_test <- fullTest %>%
    filter(Store==store)
  
  for(dept in unique(store_test$Dept)){
    
    ## Filter Test and Training Data
    dept_train <- store_train %>%
      filter(Dept==dept)
    dept_test <- store_test %>%
      filter(Dept==dept)
    
    ## If Statements for data scenarios
    if(nrow(dept_train)==0){
      
      ## Predict 0
      preds <- dept_test %>%
        transmute(Id=paste(Store, Dept, Date, sep="_"),
               Weekly_Sales=0)
      
    } else if(nrow(dept_train) < 10 && nrow(dept_train) > 0){
      
      ## Predict the mean
      preds <- dept_test %>%
        transmute(Id=paste(Store, Dept, Date, sep="_"),
                  Weekly_Sales=mean(dept_train$Weekly_Sales))
      
    } else {
      
      ## Fit a penalized regression model
      my_recipe <- recipe(Weekly_Sales ~ ., data = dept_train) %>%
        step_mutate(Holiday = as.integer(IsHoliday)) %>%
        step_date(Date, features=c("month","year")) %>%
        step_rm(Date, Store, Dept, IsHoliday)
      prepped_recipe <- prep(my_recipe)
      tst <- bake(prepped_recipe, new_data=dept_test)
      
      my_model <- rand_forest(mtry=3,
                              trees=100,
                              min_n=5) %>%
        set_engine("ranger") %>%
        set_mode("regression")
      
      my_wf <- workflow() %>%
        add_recipe(my_recipe) %>%
        add_model(my_model) %>%
        fit(dept_train)

      preds <- dept_test %>%
        transmute(Id=paste(Store, Dept, Date, sep="_"),
                  Weekly_Sales=predict(my_wf, new_data = .) %>%
                    pull(.pred))
      
    }
  
    ## Bind predictions together
    all_preds <- bind_rows(all_preds,
                           preds)
    
    ## Print out Progress
    cntr <- cntr+1
    cat("Store", store, "Department", dept, "Completed.",
        round(100 * cntr / n_storeDepts, 1), "% overall complete.\n")
    
  } ## End Dept Loop
  
} ## End Store Loop

## Write out after each store so I don't have to start over
vroom_write(x=all_preds, 
            file=paste0("./Predictions.csv"), delim=",")



