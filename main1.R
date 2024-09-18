# Load necessary libraries
library(caret)
library(ggplot2)
library(dplyr)
library(randomForest)  # For Random Forest
library(DAAG) # for VIF

# Convert specific categorical variables to factors
selected_data$Vertical_Segment <- as.factor(selected_data$Vertical_Segment)
selected_data$Product_Collection <- as.factor(selected_data$Product_Collection)
selected_data$Intel_Hyper_Threading_Technology_ <- as.factor(selected_data$Intel_Hyper_Threading_Technology_)
numerical_data <- selected_data %>% select(where(is.numeric))

# Define performance metrics functions
performance_metrics <- function(predictions, actual) {
  mse <- mean((predictions - actual)^2)
  r_squared <- 1 - (sum((actual - predictions)^2, na.rm = TRUE) / sum((actual - mean(actual))^2, na.rm = TRUE))
  
  return(list(
    R_squared = r_squared,
    MSE = mse
  ))
}

# Set seed for reproducibility
set.seed(123)

# Split data into training and testing sets (80-20 split)
trainIndex <- createDataPartition(selected_data$Processor_Base_Frequency, p = 0.8, list = FALSE)
trainData <- selected_data[trainIndex, ]
testData <- selected_data[-trainIndex, ]

# Train linear regression model
linear_model <- lm(Processor_Base_Frequency ~ ., data = trainData)
linear_predictions <- predict(linear_model, newdata = testData)

# Calculate residuals for linear model
linear_residuals <- testData$Processor_Base_Frequency - linear_predictions

# Store performance metrics for linear model
linear_metrics <- performance_metrics(linear_predictions, testData$Processor_Base_Frequency)

# Plot predicted vs actual for Linear Regression Model
pred_vs_actual_plot_linear <- ggplot() +
  geom_point(data = testData, aes(x = Processor_Base_Frequency, y = linear_predictions), color = "blue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "Actual Processor Base Frequency", y = "Predicted Processor Base Frequency",
       title = "Predicted vs Actual Processor Base Frequency for Linear Regression") +
  theme_minimal()

# Residuals vs. Fitted Values Plot for Linear Regression Model
residuals_vs_fitted_linear <- ggplot() +
  geom_point(aes(x = linear_predictions, y = linear_residuals), color = "blue") +
  geom_smooth(method = "loess", color = "red") +
  labs(x = "Fitted Values", y = "Residuals",
       title = "Residuals vs. Fitted Values for Linear Regression Model") +
  theme_minimal()

# Plot residuals distribution for Linear Regression Model
residuals_distribution_plot_linear <- ggplot() +
  geom_histogram(aes(x = linear_residuals), bins = 30, fill = "blue", color = "black", alpha = 0.5) +
  labs(x = "Residuals", y = "Frequency",
       title = "Distribution of Residuals for Linear Regression Model") +
  theme_minimal()

print(residuals_distribution_plot_linear)
# Calculate VIF values
vif_values <- vif(linear_model)

# Print VIF values
print(vif_values)

# Train Random Forest model
rf_model <- randomForest(Processor_Base_Frequency ~ ., data = trainData, importance = TRUE)
rf_predictions <- predict(rf_model, newdata = testData)

# Calculate residuals for Random Forest model
rf_residuals <- testData$Processor_Base_Frequency - rf_predictions

# Store performance metrics for Random Forest
rf_metrics <- performance_metrics(rf_predictions, testData$Processor_Base_Frequency)

# Print performance metrics for linear regression
cat("\nPerformance Metrics for Linear Regression Model:\n")
print(linear_metrics)

# Print performance metrics for Random Forest
cat("\nPerformance Metrics for Random Forest Model:\n")
print(rf_metrics)


# Plot predicted vs actual for Random Forest Model
pred_vs_actual_plot_rf <- ggplot() +
  geom_point(data = testData, aes(x = Processor_Base_Frequency, y = rf_predictions), color = "green", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "Actual Processor Base Frequency", y = "Predicted Processor Base Frequency",
       title = "Predicted vs Actual Processor Base Frequency for Random Forest") +
  theme_minimal()

print(pred_vs_actual_plot_rf)


# Plot residuals distribution for Random Forest Model
residuals_distribution_plot_rf <- ggplot() +
  geom_histogram(aes(x = rf_residuals), bins = 30, fill = "green", color = "black", alpha = 0.5) +
  labs(x = "Residuals", y = "Frequency",
       title = "Distribution of Residuals for Random Forest Model") +
  theme_minimal()

print(residuals_distribution_plot_rf)



