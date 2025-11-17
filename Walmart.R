# load packages

library(tidyverse)
library(tidymodels)

# load data

test <- read_csv('test.csv', show_col_types = F)
train <- read_csv('train.csv', show_col_types = F)
features <- read_csv('features.csv', show_col_types = F)


# pre-prosessing

features <- features %>%
  mutate(across(starts_with("Markdown"), ~ replace_na(., 0))) %>%
  mutate(TotalMarkdown = rowSums(across(starts_with("Markdown")))) %>%
  mutate(MarkdownFlag = if_else(TotalMarkdown > 0, 1, 0))%>%
  select(-c(MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5))

train <- train %>%
  left_join(features, by = "Store", "Date", relationship = "many-to-many")

test <- test %>%
  left_join(features, by = "Store", "Date", relationship = "many-to-many")