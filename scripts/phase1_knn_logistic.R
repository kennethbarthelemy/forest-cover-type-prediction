#QTM 7571 Project:Phase 1
#Predicting forest species composition from satellite data Links to an external site
#Group 4: Kenneth Barthelemy, Dilo Junior, Javier Martinez, Priyanka Tambe


## 1. Library Import---------------------------------------------------------------------------------
## In this section, we load all necessary R packages and external scripts to perform data analysis and modeling. Keeping imports organized at the top ensures reproducibility and clarity.
library(rpart)
library(rpart.plot)
library(ggplot2)
library(reshape2)
library(caret)
library(nnet)
source("C:/Users/javim/OneDrive/Babson Courses/Electives/Machine Learning Meathods/data/BabsonAnalytics.R")



## 2. Load Data into df -------------------------------------------------------------------------------
## Here we load the dataset from a CSV file into a dataframe named df. This dataframe serves as the basis for all further analysis and modeling.
df = read.csv("./data/train.csv")  # Load training data from data folder
summary(df)



## 3. Data Exploration----------------------------------------------------------------------------------
## This section checks the quality and structure of the dataset, specifically looking for missing values and verifying assumptions about categorical data (e.g., ensuring each data point has exactly one soil and wilderness type).

# 3.a Checking Null/NA
#Several ways of checking for any NA/NULL
anyNA(df)
missing_counts = colSums(is.na(df))
missing_counts

# 3.b Checking Sum Across Wilderness Area & Soil Types
#Wilderness Area Check: Checking that every row has only one unique wilderness_area
WA_cols = paste0("Wilderness_Area", 1:4) #Creating a list of Soil_Type columns 1-40
WA_sums = rowSums(df[, WA_cols]) #WA 1-4 sum across these columns for each row
WA_unique_sums = unique(WA_sums)  #Check the unique sums
print(WA_unique_sums)
#Soil Type Check: Checking that every row has only one unique soil type
soil_cols = paste0("Soil_Type", 1:40) #Creating a list of Soil_Type columns 1-40
soil_sums = rowSums(df[, soil_cols]) #Soil 1-40 sum across these columns for each row
soil_unique_sums = unique(soil_sums)  #Check the unique sums
print(soil_unique_sums)  #Since this number is 1, we know that every id, has only 1 soil type attributed to it
# Soil Type Check: Alternate Check, verify if every row sums to exactly 1:
all_single_soil = all(soil_sums == 1)
print(all_single_soil)  # Returns TRUE since each 30x30 grid has only one soil type



## 4. Data Management----------------------------------------------------------------------------------
## In this section, we clean, transform, and simplify the dataset here to prepare it for modeling. This involves removing unnecessary columns (like IDs), converting categorical data into more readable formats, and consolidating multiple binary indicator columns into single categorical variables.

# 4.a Drop ID as it will give no useful insight for the models
df$Id = NULL

# 4.b Transform Cover_Type (Target) into factor
df$Cover_Type = as.factor(df$Cover_Type)

# 4.c Consolidate columns Wilderness_Area 1-4 and Soil_Type 1-40 into a single column and drop the binary columns
# Define the column names for wilderness and soil type
wilderness_cols = paste0("Wilderness_Area", 1:4)
soil_cols = paste0("Soil_Type", 1:40)  # Assumes columns are named Soil_Type1, Soil_Type2, ..., Soil_Type40

# 4.d Consolidate wilderness & Soil columns into a single factor column
df$Wilderness_Area = factor(max.col(df[, wilderness_cols]), levels = 1:4)
df$Soil_Type = factor(max.col(df[, soil_cols]), levels = 1:40)

# 4.e Drop the original binary columns
df = df[, !(names(df) %in% c(wilderness_cols, soil_cols))]

# 4.f Keep Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points as Int

## 5. Data Visualization----------------------------------------------------------------------------------
## This Section helps us understand the data visually through plots such as bar charts, histograms, and box plots. These visualizations help identify meaningful patterns, distributions, and relationships between variables.

# 5.a Frequency Table: Count how many observations fall into each cover type
cover_distribution = table(df$Cover_Type)
print(cover_distribution)

barplot(cover_distribution,
        main = "Distribution of Cover Types",
        xlab = "Cover Type",
        ylab = "Frequency",
        col = "skyblue")

ggplot(df, aes(x = factor(Cover_Type))) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Distribution of Cover Types", x = "Cover Type", y = "Frequency") +
  theme_minimal()

# 5.b Box-Plots for Continuous Variables
cont_vars <- c("Elevation", "Aspect", "Slope", 
               "Horizontal_Distance_To_Hydrology", 
               "Vertical_Distance_To_Hydrology",
               "Horizontal_Distance_To_Roadways", 
               "Hillshade_9am", "Hillshade_Noon", 
               "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points")

df_long_box = melt(df, id.vars = "Cover_Type", measure.vars = cont_vars) #Reshape the data from wide to long format using melt()

ggplot(df_long_box, aes(x = factor(Cover_Type), y = value, fill = factor(Cover_Type))) + #Create box plots by variable, with Cover_Type on the X axis and different fill colors
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +
  labs(title = "Boxplots of Continuous Variables by Cover Type",
       x = "Cover Type",
       y = "Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")


# 5.c Create faceted histograms for each continuous variable
df_long_hist = melt(df, measure.vars = cont_vars)

ggplot(df_long_hist, aes(x = value)) +    #Create histograms by variable
  geom_histogram(bins = 30, fill = "lightblue", color = "black") + 
  facet_wrap(~ variable, scales = "free_x", ncol = 3) +  
  labs(title = "Histograms of Continuous Variables",
       x = "Value",
       y = "Count") +
  theme_minimal()



# 6 Train-Test Split Partition: 60-40 --------------------------------------------------------------
## In this section, we divide our dataset into two parts: a training set used to build models, and a test set used to evaluate the accuracy and reliability of our models

N = nrow(df)
training_size = round(N*0.6)
training_cases = sample(N, training_size)
training = df[training_cases,]
test = df[-training_cases,]

## 7 Model 1: KNN  ---------------------------------------------------------------------
## In this section, we build our first predictive model using K-Nearest Neighbors (KNN). We preprocess the data specifically for KNN, standardizing numeric variables, choosing the optimal number of neighbors (K), and evaluating the model’s error rate.

# 7.a Data Cleaning
df_knn = df #Need to create a Dataframe with only numerical values for KNN. We will first create a copy then drop the non-numerical values
df_knn$Wilderness_Area = NULL
df_knn$Soil_Type = NULL

# 7.b Standardize numerical data
standardizer = preProcess(df_knn, method=c("center","scale"))
df_knn = predict(standardizer,df_knn)

# 7.c Train-Test Partition
knn_training = df_knn[training_cases,] #Taking the same rows as training to compare performance with model 2
knn_test = df_knn[-training_cases,]

# 7.d Creating model with K=5
knn_model = knn3(Cover_Type ~ ., data = knn_training, k=5)

# 7.e Predictions
knn_predictions = predict(knn_model, knn_test, type="class")

# 7.f Evaluate
knn_observations = knn_test$Cover_Type
knn_error_rate = sum(knn_predictions != knn_observations)/nrow(knn_test)
knn_table = table(knn_predictions, knn_observations)
knn_table

# 7.g Bench
error_bench = benchmarkErrorRate(knn_training$Cover_Type, knn_test$Cover_Type)


## 8. Model 2: Logistic Regression ----------------------------------------------------------------
## Our second predictive model uses multinomial logistic regression. This section shows building and refining the logistic regression model through stepwise selection of variables, making predictions, and assessing the model's error rate

# 8.a Creating model 
lr_model = multinom(Cover_Type ~ ., data = training)
#summary(lr_model)

# 8.b Controling for overfit with stepwise removal of variables
step_model = step(lr_model)
#summary(step_model)

# 8.c Evaluate
lr_predictions = predict(step_model,test)
lr_observations = test$Cover_Type
lr_error_rate = sum(lr_predictions != lr_observations)/nrow(test)
table(lr_predictions,lr_observations)

error_bench = benchmarkErrorRate(training$Cover_Type, test$Cover_Type) #is same as KNN bench, but double checking



