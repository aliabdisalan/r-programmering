# Install necessary packages (assuming they are not already installed)
if (!require("pxweb")) install.packages("pxweb")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("caret")) install.packages("caret")
if (!require("randomForest")) install.packages("randomForest")
if (!require("e1071")) install.packages("e1071")

library(pxweb)
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)

# Define PX-WEB query parameters
pxweb_query_list <- list(
  "Region" = c("*"),
  "Agarkategori" = c("*"),
  "ContentsCode" = c("TK1001AB"),
  "Tid" = c("*")
)

# Download data using pxweb_query and pxweb_get
pxq <- pxweb_query(pxweb_query_list)
pxd <- pxweb_get(
  "https://api.scb.se/OV0104/v1/doris/en/ssd/TK/TK1001/TK1001A/PersBilarA",
  pxq
)

# Convert to data frame and handle variable value types appropriately
pxdf <- as.data.frame(pxd, column.name.type = "text", variable.value.type = "text")
pxdf
# Data cleaning and pre-processing
# - Check for missing values and handle them (e.g., impute or remove rows)
# - Explore data distribution and consider transformations if necessary (e.g., log transformation)
# - Feature engineering (create new features if relevant)

# Rename columns
colnames(pxdf) <- c("Region", "Ownership_Category", "Year", "Number_of_Passenger_Cars")
head(pxdf)

str(pxdf)

# to see the number of unique values in the variable.
nlevels(pxdf$Number_of_Passenger_Cars)

pxdf$Number_of_Passenger_Cars

# Convert Year to numeric
pxdf$Year <- as.numeric(pxdf$Year)

# Remove rows with missing values
pxdf <- pxdf[complete.cases(pxdf$Number_of_Passenger_Cars), ]

# Split data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(pxdf$Number_of_Passenger_Cars, p = 0.8, list = FALSE)
train_data <- pxdf[train_index, ]
test_data <- pxdf[-train_index, ]

# Linear regression model (consider adding interaction terms or non-linear features)
lm_model <- lm(Number_of_Passenger_Cars ~ Year + I(Year^2), data = train_data)  # Include potential non-linearity with Year^2
summary(lm_model)

# Make predictions on test data
predictions <- predict(lm_model, newdata = test_data)

# Evaluate model performance
RMSE <- sqrt(mean((predictions - test_data$Number_of_Passenger_Cars)^2))
cat("Linear Regression Root Mean Squared Error:", RMSE, "\n")

# Define a threshold for acceptable deviation
threshold <- 0.1  # for example, consider predictions within 0.1 of actual values as accurate

# Calculate accuracy
accuracy <- mean(abs(predictions - test_data$Number_of_Passenger_Cars) <= threshold)
accuracy_percentage <- accuracy * 100

cat("Accuracy:", accuracy_percentage, "%\n")

# Visualize the model
plot(test_data$Year, test_data$Number_of_Passenger_Cars, col = "blue", pch = 16,
     xlab = "Year", ylab = "Number of Passenger Cars", main = "Linear Regression Model")
lines(test_data$Year, predictions, col = "red", lwd = 2)
legend("topleft", legend = c("Observed", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)

# Random Forest model (tune hyperparameters for potentially better performance)
rf_model <- randomForest(Number_of_Passenger_Cars ~ Year, data = train_data, ntree = 500)  # Increased number of trees (adjust as needed)
# Use caret's train function for hyperparameter tuning (consider exploring different parameters)
# ctrl <- trainControl(method = "cv")
# rf_model <- train(Number_of_Passenger_Cars ~ Year, data = train_data, method = "rf", trControl = ctrl)

# Make predictions on test data
rf_predictions <- predict(rf_model, newdata = test_data)

# Evaluate model performance
rf_RMSE <- sqrt(mean((rf_predictions - test_data$Number_of_Passenger_Cars)^2))
cat("Random Forest Root Mean Squared Error:", rf_RMSE, "\n")

# Define a threshold for acceptable deviation
threshold <- 0.1  # for example, consider predictions within 0.1 of actual values as accurate

# Calculate accuracy
accuracy <- mean(abs(rf_predictions - test_data$Number_of_Passenger_Cars) <= threshold)
accuracy_percentage <- accuracy * 100

cat("Accuracy RF:", accuracy_percentage, "%\n")


# Visualize the model
plot(test_data$Year, test_data$Number_of_Passenger_Cars, col = "blue", pch = 16,
     xlab = "Year", ylab = "Number of Passenger Cars", main = "Random Forest Model")
lines(test_data$Year, rf_predictions, col = "red", lwd = 2)
legend("topleft", legend = c("Observed", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)

# SVM regression model (tune hyperparameters for potentially better performance)
svm_model <- svm(Number_of_Passenger_Cars ~ Year, data = train_data, kernel = "linear")  # Start with linear kernel (adjust as needed)
# Use caret's train function for hyperparameter tuning (consider exploring different parameters)
# ctrl <- trainControl(method = "cv")
# svm_model <- train(Number_of_Passenger_Cars ~ Year, data = train_data, method = "svm", trControl = ctrl)

# Make predictions on test data
svm_predictions <- predict(svm_model, newdata = test_data)

# Evaluate model performance
svm_RMSE <- sqrt(mean((svm_predictions - test_data$Number_of_Passenger_Cars)^2))
cat("SVM Root Mean Squared Error:", svm_RMSE, "\n")

# Visualize the model
plot(test_data$Year, test_data$Number_of_Passenger_Cars, col = "blue", pch = 16,
     xlab = "Year", ylab = "Number of Passenger Cars", main = "SVM regression model")
lines(test_data$Year, svm_predictions, col = "red", lwd = 2)
legend("topleft", legend = c("Observed", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)




