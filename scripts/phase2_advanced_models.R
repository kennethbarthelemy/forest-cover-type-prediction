# QTM 7571 Project Phase 2
# Predicting forest species composition from satellite data
# Group 4: Kenneth Barthelemy, Dilo Junior, Javier Martinez, Priyanka Tambe


# 1. Load Libraries

library(rpart)
library(rpart.plot)
library(ggplot2)
library(reshape2)
library(caret)
library(nnet)
library(randomForest)
library(class)
source("C:/Users/javim/OneDrive/Babson Courses/Electives/Machine Learning Meathods/data/BabsonAnalytics.R")

# 2. Loading Data

df = read.csv("./data/train.csv")  # Load training data from data folder
df$Id <- NULL
df$Cover_Type <- as.factor(df$Cover_Type)

# Check for NAs
anyNA(df)
colSums(is.na(df))

# Validate binary structure
wilderness_cols <- paste0("Wilderness_Area", 1:4)
soil_cols <- paste0("Soil_Type", 1:40)
stopifnot(all(rowSums(df[, wilderness_cols]) == 1))
stopifnot(all(rowSums(df[, soil_cols]) == 1))


# 3. Data Cleaning

# Combine categorical binary columns into single factor columns
df$Wilderness_Area <- factor(max.col(df[, wilderness_cols]), levels = 1:4)
df$Soil_Type <- factor(max.col(df[, soil_cols]), levels = 1:40)
df <- df[, !(names(df) %in% c(wilderness_cols, soil_cols))]

# Add engineered features
df$Total_Distance <- df$Horizontal_Distance_To_Hydrology + 
  abs(df$Vertical_Distance_To_Hydrology) + 
  df$Horizontal_Distance_To_Roadways + 
  df$Horizontal_Distance_To_Fire_Points

df$Slope_Elevation <- df$Slope * df$Elevation

# Standardize numeric variables
cont_vars <- c("Elevation", "Aspect", "Slope", 
               "Horizontal_Distance_To_Hydrology", 
               "Vertical_Distance_To_Hydrology",
               "Horizontal_Distance_To_Roadways", 
               "Hillshade_9am", "Hillshade_Noon", 
               "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
               "Total_Distance", "Slope_Elevation")

standardizer <- preProcess(df[, cont_vars], method = c("center", "scale"))
df[, cont_vars] <- predict(standardizer, df[, cont_vars])


# 4. Data visualization

df_long_hist <- melt(df, measure.vars = cont_vars)
ggplot(df_long_hist, aes(x = value)) +
  geom_histogram(bins = 30, fill = "lightblue", color = "black") +
  facet_wrap(~ variable, scales = "free_x", ncol = 3) +
  labs(title = "Histograms of Continuous Variables", x = "Value", y = "Count") +
  theme_minimal()


cont_vars <- c("Elevation", "Aspect", "Slope", 
               "Horizontal_Distance_To_Hydrology", 
               "Vertical_Distance_To_Hydrology",
               "Horizontal_Distance_To_Roadways", 
               "Hillshade_9am", "Hillshade_Noon", 
               "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points")

df_long_box = melt(df, id.vars = "Cover_Type", measure.vars = cont_vars) 

ggplot(df_long_box, aes(x = factor(Cover_Type), y = value, fill = factor(Cover_Type))) + 
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +
  labs(title = "Boxplots of Continuous Variables by Cover Type",
       x = "Cover Type",
       y = "Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")


# 5. Train-Test Split

N <- nrow(df)
train_size <- round(N * 0.6)
train_idx <- sample(N, train_size)
training <- df[train_idx, ]
test <- df[-train_idx, ]
observations = test$Cover_Type

# --------------------------------------------------------------------------------
# 6. Model 1: K-Nearest Neighbors
# --------------------------------------------------------------------------------
df_knn <- df
knn_vars <- df_knn[, !(names(df_knn) %in% c("Wilderness_Area", "Soil_Type"))]
knn_train <- knn_vars[train_idx, ]
knn_test <- knn_vars[-train_idx, ]
knn_k <- round(sqrt(N))
knn_model <- knn3(Cover_Type ~ ., data = knn_train, k = knn_k)
knn_pred <- predict(knn_model, knn_test, type = "class")
knn_error <- mean(knn_pred != knn_test$Cover_Type)

table(knn_pred,observations)

# --------------------------------------------------------------------------------
# 7. Model 2: Logistic Regression
# --------------------------------------------------------------------------------
lr_model <- multinom(Cover_Type ~ ., data = training)
lr_step <- step(lr_model)
lr_pred <- predict(lr_step, test)
lr_error <- mean(lr_pred != test$Cover_Type)

table(lr_pred,observations)
coef(lr_model)[,1:10 ]


# --------------------------------------------------------------------------------
# 8. Model 3: Random Forest
# --------------------------------------------------------------------------------
rf_model <- randomForest(Cover_Type ~ ., data = training, ntree = 100, importance = TRUE)
rf_pred <- predict(rf_model, test)
rf_error <- mean(rf_pred != test$Cover_Type)

table(rf_pred,observations)
importance(rf_model)

# --------------------------------------------------------------------------------
# 9. Ensemble: Stacking
# --------------------------------------------------------------------------------
knn_full <- predict(knn_model, df, type = "class")
lr_full <- predict(lr_step, df)
rf_full <- predict(rf_model, df)

df_stack <- cbind(df,knn_full, lr_full, rf_full)

standardizer = preProcess(df_stack, method=c("center","scale"))
df_stack = predict(standardizer, df_stack)

training_stack = df_stack[train_idx, ]
test_stack = df_stack[-train_idx, ]


stack_model <- nnet(Cover_Type ~ ., data = training_stack, size=6)
stack_pred <- predict(stack_model, test_stack)
stack_pred <- max.col(stack_pred)
stack_error <- mean(stack_pred != test$Cover_Type)

table(stack_pred,observations)


# --------------------------------------------------------------------------------
# 10. Final Model Comparison
# --------------------------------------------------------------------------------
cat("\nMODEL PERFORMANCE COMPARISON\n")
cat("-------------------------------------------------\n")
cat(sprintf("1. KNN (k = 123):               Error Rate = %.4f\n", knn_error))
cat(sprintf("2. Logistic Regression:       Error Rate = %.4f\n", lr_error))
cat(sprintf("3. Random Forest (100 trees): Error Rate = %.4f\n", rf_error))
cat(sprintf("4. Stacked Model:             Error Rate = %.4f\n", stack_error))
cat("-------------------------------------------------\n")
