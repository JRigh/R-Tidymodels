#--------------------------
# R Tidymodels introduction 
#--------------------------

library("conflicted")
library("tidymodels")

# Additional packages for dataviz etc.
library("ggrepel")     # for geom_label_repel()
library("corrplot")    # for corrplot()
#> corrplot 0.84 loaded

conflict_prefer("filter", "dplyr")
ggplot2::theme_set(theme_light())

# Diamonds dataset: correlations
data("diamonds")
diamonds %>%
  sample_n(2000) %>% 
  mutate_if(is.factor, as.numeric) %>%
  select(price, everything()) %>%
  cor %>%
  {.[order(abs(.[, 1]), decreasing = TRUE), 
     order(abs(.[, 1]), decreasing = TRUE)]} %>%
  corrplot(method = "number", type = "upper", mar = c(0, 0, 1.5, 0),
           title = "Correlations between price and various features of diamonds")

# split dataset into training and testing set
set.seed(1243)

dia_split <- initial_split(diamonds, prop = .1, strata = price)

dia_train <- training(dia_split)
dia_test  <- testing(dia_split)

dim(dia_train)
#> [1] 5395   10
dim(dia_test)
#> [1] 48545    10

# the training data set is prepared for 3-fold cross-validation
dia_vfold <- vfold_cv(dia_train, v = 3, repeats = 1, strata = price)
dia_vfold %>% 
  mutate(df_ana = map(splits, analysis),
         df_ass = map(splits, assessment))

# Data visualization

qplot(carat, price, data = dia_train) +
  scale_y_continuous(trans = log_trans(), labels = function(x) round(x, -2)) +
  geom_smooth(method = "lm", formula = "y ~ poly(x, 4)") +
  labs(title = "Nonlinear relationship between price and carat of diamonds",
       subtitle = "The degree of the polynomial is a potential tuning parameter")

# prepare the recipe: log transform the outcome 'price', add a quadratic effect to carat
# center and scale the numeric predictors
dia_rec <-
  recipe(price ~ ., data = dia_train) %>%
  step_log(all_outcomes()) %>%
  step_normalize(all_predictors(), -all_nominal()) %>%
  step_dummy(all_nominal()) %>%
  step_poly(carat, degree = 2)

prep(dia_rec)

# Note the linear and quadratic term for carat and the dummies for e.g. color
dia_juiced <- juice(prep(dia_rec))
dim(dia_juiced)
#> [1] 5395   25
names(dia_juiced)

# create a linear model
lm_model <-
  linear_reg() %>%
  set_mode("regression") %>%
  set_engine("lm")

# fit a random forest model
rand_forest(mtry = 3, trees = 500, min_n = 5) %>%
  set_mode("regression") %>%
  set_engine("ranger", importance = "impurity_corrected")

# fit the linear model
lm_fit1 <- fit(lm_model, price ~ ., dia_juiced)
lm_fit1

# model summary
glance(lm_fit1$fit)
tidy(lm_fit1) %>% 
  arrange(desc(abs(statistic)))

# predictions and residuals
lm_predicted <- augment(lm_fit1$fit, data = dia_juiced) %>% 
  rowid_to_column()
select(lm_predicted, rowid, price, .fitted:.std.resid)

ggplot(lm_predicted, aes(.fitted, price)) +
  geom_point(alpha = .2) +
  ggrepel::geom_label_repel(aes(label = rowid), 
                            data = filter(lm_predicted, abs(.resid) > 2)) +
  labs(title = "Actual vs. Predicted Price of Diamonds")

# How does dia_vfold look?
dia_vfold
#> #  3-fold cross-validation using stratification 
#> # A tibble: 3 x 2
#>   splits              id   
#>   <named list>        <chr>
#> 1 <split [3.6K/1.8K]> Fold1
#> 2 <split [3.6K/1.8K]> Fold2
#> 3 <split [3.6K/1.8K]> Fold3

# Extract analysis/training and assessment/testing data
lm_fit2 <- mutate(dia_vfold,
                  df_ana = map (splits,  analysis),
                  df_ass = map (splits,  assessment))
lm_fit2
#> #  3-fold cross-validation using stratification 
#> # A tibble: 3 x 4
#>   splits              id    df_ana                df_ass               
#> * <named list>        <chr> <named list>          <named list>         
#> 1 <split [3.6K/1.8K]> Fold1 <tibble [3,596 x 10]> <tibble [1,799 x 10]>
#> 2 <split [3.6K/1.8K]> Fold2 <tibble [3,596 x 10]> <tibble [1,799 x 10]>
#> 3 <split [3.6K/1.8K]> Fold3 <tibble [3,598 x 10]> <tibble [1,797 x 10]>

lm_fit2 <- 
  lm_fit2 %>% 
  # prep, juice, bake
  mutate(
    recipe = map (df_ana, ~prep(dia_rec, training = .x)),
    df_ana = map (recipe,  juice),
    df_ass = map2(recipe, 
                  df_ass, ~bake(.x, new_data = .y))) %>% 
  # fit
  mutate(
    model_fit  = map(df_ana, ~fit(lm_model, price ~ ., data = .x))) %>% 
  # predict
  mutate(
    model_pred = map2(model_fit, df_ass, ~predict(.x, new_data = .y)))

select(lm_fit2, id, recipe:model_pred)
#> # A tibble: 3 x 4
#>   id    recipe       model_fit    model_pred          
#>   <chr> <named list> <named list> <named list>        
#> 1 Fold1 <recipe>     <fit[+]>     <tibble [1,799 x 1]>
#> 2 Fold2 <recipe>     <fit[+]>     <tibble [1,799 x 1]>
#> 3 Fold3 <recipe>     <fit[+]>     <tibble [1,797 x 1]>

# cross-validation predictions
lm_preds <- 
  lm_fit2 %>% 
  mutate(res = map2(df_ass, model_pred, ~data.frame(price = .x$price,
                                                    .pred = .y$.pred))) %>% 
  select(id, res) %>% 
  tidyr::unnest(res) %>% 
  group_by(id)
lm_preds
#> # A tibble: 5,395 x 3
#> # Groups:   id [3]
#>    id    price .pred
#>    <chr> <dbl> <dbl>
#>  1 Fold1  5.84  5.83
#>  2 Fold1  6.00  6.25
#>  3 Fold1  6.00  6.05
#>  4 Fold1  6.32  6.56
#>  5 Fold1  6.32  6.31
#>  6 Fold1  7.92  7.73
#>  7 Fold1  7.93  7.58
#>  8 Fold1  7.93  7.80
#>  9 Fold1  7.93  7.88
#> 10 Fold1  7.94  7.91
#> # ... with 5,385 more rows

metrics(lm_preds, truth = price, estimate = .pred)
#> # A tibble: 9 x 4
#>   id    .metric .estimator .estimate
#>   <chr> <chr>   <chr>          <dbl>
#> 1 Fold1 rmse    standard       0.168
#> 2 Fold2 rmse    standard       0.147
#> 3 Fold3 rmse    standard       0.298
#> 4 Fold1 rsq     standard       0.973
#> 5 Fold2 rsq     standard       0.979
#> 6 Fold3 rsq     standard       0.918
#> 7 Fold1 mae     standard       0.116
#> 8 Fold2 mae     standard       0.115
#> 9 Fold3 mae     standard       0.110

# hyperparameter tuning for random forest
rf_model <- 
  rand_forest(mtry = tune()) %>%
  set_mode("regression") %>%
  set_engine("ranger")

parameters(rf_model)
#> Collection of 1 parameters for tuning
#> 
#>    id parameter type object class
#>  mtry           mtry    nparam[?]
#> 
#> Model parameters needing finalization:
#>    # Randomly Selected Predictors ('mtry')
#> 
#> See `?dials::finalize` or `?dials::update.parameters` for more information.
mtry()
#> # Randomly Selected Predictors  (quantitative)
#> Range: [1, ?]

rf_model %>% 
  parameters() %>% 
  update(mtry = mtry(c(1L, 5L)))
#> Collection of 1 parameters for tuning
#> 
#>    id parameter type object class
#>  mtry           mtry    nparam[+]

rf_model %>% 
  parameters() %>% 
  # Here, the maximum of mtry equals the number of predictors, i.e., 24.
  finalize(x = select(juice(prep(dia_rec)), -price)) %>% 
  pull("object")
#> [[1]]
#> # Randomly Selected Predictors  (quantitative)
#> Range: [1, 24]


# Note that this recipe cannot be prepped (and juiced), since "degree" is a
# tuning parameter
dia_rec2 <-
  recipe(price ~ ., data = dia_train) %>%
  step_log(all_outcomes()) %>%
  step_normalize(all_predictors(), -all_nominal()) %>%
  step_dummy(all_nominal()) %>%
  step_poly(carat, degree = tune())

dia_rec2 %>% 
  parameters() %>% 
  pull("object")
#> [[1]]
#> Polynomial Degree  (quantitative)
#> Range: [1, 3]

rf_wflow <-
  workflow() %>%
  add_model(rf_model) %>%
  add_recipe(dia_rec2)
rf_wflow
#> == Workflow ==============================================================================
#> Preprocessor: Recipe
#> Model: rand_forest()
#> 
#> -- Preprocessor --------------------------------------------------------------------------
#> 4 Recipe Steps
#> 
#> * step_log()
#> * step_normalize()
#> * step_dummy()
#> * step_poly()
#> 
#> -- Model ---------------------------------------------------------------------------------
#> Random Forest Model Specification (regression)
#> 
#> Main Arguments:
#>   mtry = tune()
#> 
#> Computational engine: ranger

rf_param <-
  rf_wflow %>%
  parameters() %>%
  update(mtry = mtry(range = c(3L, 5L)),
         degree = degree_int(range = c(2L, 4L)))
rf_param$object
#> [[1]]
#> # Randomly Selected Predictors  (quantitative)
#> Range: [3, 5]
#> 
#> [[2]]
#> Polynomial Degree  (quantitative)
#> Range: [2, 4]


rf_grid <- grid_regular(rf_param, levels = 3)
rf_grid
#> # A tibble: 9 x 2
#>    mtry degree
#>   <int>  <int>
#> 1     3      2
#> 2     4      2
#> 3     5      2
#> 4     3      3
#> 5     4      3
#> 6     5      3
#> 7     3      4
#> 8     4      4
#> 9     5      4

library("doFuture")
all_cores <- parallel::detectCores(logical = FALSE) - 1

registerDoFuture()
cl <- makeCluster(all_cores)
plan(future::cluster, workers = cl)
.

rf_search <- tune_grid(rf_wflow, grid = rf_grid, resamples = dia_vfold,
                       param_info = rf_param)


autoplot(rf_search, metric = "rmse") +
  labs(title = "Results of Grid Search for Two Tuning Parameters of a Random Forest")

show_best(rf_search, "rmse", n = 9)
#> # A tibble: 9 x 7
#>    mtry degree .metric .estimator  mean     n std_err
#>   <int>  <int> <chr>   <chr>      <dbl> <int>   <dbl>
#> 1     5      2 rmse    standard   0.121     3 0.00498
#> 2     5      3 rmse    standard   0.121     3 0.00454
#> 3     4      2 rmse    standard   0.122     3 0.00463
#> 4     5      4 rmse    standard   0.122     3 0.00471
#> 5     4      3 rmse    standard   0.123     3 0.00469
#> 6     4      4 rmse    standard   0.124     3 0.00496
#> 7     3      3 rmse    standard   0.128     3 0.00502
#> 8     3      2 rmse    standard   0.128     3 0.00569
#> 9     3      4 rmse    standard   0.128     3 0.00501

select_best(rf_search, metric = "rmse")
#> # A tibble: 1 x 2
#>    mtry degree
#>   <int>  <int>
#> 1     5      2

select_by_one_std_err(rf_search, mtry, degree, metric = "rmse")
#> # A tibble: 1 x 9
#>    mtry degree .metric .estimator  mean     n std_err .best .bound
#>   <int>  <int> <chr>   <chr>      <dbl> <int>   <dbl> <dbl>  <dbl>
#> 1     4      2 rmse    standard   0.122     3 0.00463 0.121  0.126

rf_param_final <- select_by_one_std_err(rf_search, mtry, degree,
                                        metric = "rmse")

rf_wflow_final <- finalize_workflow(rf_wflow, rf_param_final)

rf_wflow_final_fit <- fit(rf_wflow_final, data = dia_train)


dia_rec3     <- pull_workflow_prepped_recipe(rf_wflow_final_fit)
rf_final_fit <- pull_workflow_fit(rf_wflow_final_fit)

dia_test$.pred <- predict(rf_final_fit, 
                          new_data = bake(dia_rec3, dia_test))$.pred
dia_test$logprice <- log(dia_test$price)

metrics(dia_test, truth = logprice, estimate = .pred)
#> # A tibble: 3 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard      0.113 
#> 2 rsq     standard      0.988 
#> 3 mae     standard      0.0846

#----
# end 
#----