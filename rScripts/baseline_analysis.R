# Dual-Hardware Benchmarking Baseline Analysis Workflow in R
# Statistical Learning Approach to Computer Benchmarking (CPU & GPU)

if (!dir.exists("outputs")) {
  dir.create("outputs", recursive = TRUE)
}

cat("R DUAL-HARDWARE BENCHMARKING BASELINE ANALYSIS (CPU & GPU)\n")

eval_metrics <- function(predictions, actual) {
  mse <- mean((predictions - actual)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  mae <- mean(abs(predictions - actual), na.rm = TRUE)
  r2 <- 1 - (sum((actual - predictions)^2, na.rm = TRUE) / sum((actual - mean(actual))^2, na.rm = TRUE))
  return(c(R2 = round(r2, 4), RMSE = round(rmse, 4), MAE = round(mae, 4), MSE = round(mse, 4)))
}

parse_num_r <- function(x) {
  clean_str <- sub("^[^0-9]*([0-9]+\\.[0-9]+|[0-9]+).*", "\\1", as.character(x))
  return(as.numeric(clean_str))
}

# 1. INTEL CPU BENCHMARK MODELING
cat("\n[1/2] Processing Intel CPUs Dataset...\n")
cpu_path <- ifelse(file.exists("data/raw/Intel_CPUs.csv"), "data/raw/Intel_CPUs.csv", "Intel_CPUs.csv")
cpu_raw <- read.csv(cpu_path, stringsAsFactors = FALSE)

cpu_df <- data.frame(
  Processor_Base_Frequency = parse_num_r(cpu_raw$Processor_Base_Frequency),
  nb_of_Cores = parse_num_r(cpu_raw$nb_of_Cores),
  nb_of_Threads = parse_num_r(cpu_raw$nb_of_Threads),
  Lithography = parse_num_r(cpu_raw$Lithography),
  TDP = parse_num_r(cpu_raw$TDP),
  Cache = parse_num_r(cpu_raw$Cache),
  Vertical_Segment = as.factor(cpu_raw$Vertical_Segment)
)

cpu_df <- na.omit(cpu_df)
cat(sprintf("-> Processed %d clean CPU records.\n", nrow(cpu_df)))

set.seed(123)
n_cpu <- nrow(cpu_df)
train_idx_cpu <- sample(1:n_cpu, size = 0.8 * n_cpu)
train_cpu <- cpu_df[train_idx_cpu, ]
test_cpu  <- cpu_df[-train_idx_cpu, ]

lm_cpu <- lm(Processor_Base_Frequency ~ ., data = train_cpu)
lm_preds_cpu <- predict(lm_cpu, newdata = test_cpu)
lm_perf_cpu <- eval_metrics(lm_preds_cpu, test_cpu$Processor_Base_Frequency)

cat("CPU Linear Regression Performance (R):\n")
print(lm_perf_cpu)

png("outputs/r_cpu_pred_vs_actual.png", width = 800, height = 600, res = 120)
plot(test_cpu$Processor_Base_Frequency, lm_preds_cpu,
     col = "#2b5c8f", pch = 19,
     xlab = "Actual Processor Base Frequency (GHz)",
     ylab = "Predicted Base Frequency (GHz)",
     main = "R Linear Regression: CPU Predicted vs Actual")
abline(a = 0, b = 1, col = "#d9534f", lwd = 2, lty = 2)
dev.off()

# 2. ALL GPU BENCHMARK MODELING
cat("\n[2/2] Processing All GPUs Dataset...\n")
gpu_path <- ifelse(file.exists("data/raw/All_GPUs.csv"), "data/raw/All_GPUs.csv", "All_GPUs.csv")
gpu_raw <- read.csv(gpu_path, stringsAsFactors = FALSE)

gpu_df <- data.frame(
  Core_Speed = parse_num_r(gpu_raw$Core_Speed),
  Max_Power = parse_num_r(gpu_raw$Max_Power),
  Memory = parse_num_r(gpu_raw$Memory),
  Memory_Bandwidth = parse_num_r(gpu_raw$Memory_Bandwidth),
  Process = parse_num_r(gpu_raw$Process),
  ROPs = parse_num_r(gpu_raw$ROPs),
  TMUs = parse_num_r(gpu_raw$TMUs),
  Manufacturer = as.factor(gpu_raw$Manufacturer)
)

gpu_df <- na.omit(gpu_df)
cat(sprintf("-> Processed %d clean GPU records.\n", nrow(gpu_df)))

set.seed(123)
n_gpu <- nrow(gpu_df)
train_idx_gpu <- sample(1:n_gpu, size = 0.8 * n_gpu)
train_gpu <- gpu_df[train_idx_gpu, ]
test_gpu  <- gpu_df[-train_idx_gpu, ]

lm_gpu <- lm(Core_Speed ~ ., data = train_gpu)
lm_preds_gpu <- predict(lm_gpu, newdata = test_gpu)
lm_perf_gpu <- eval_metrics(lm_preds_gpu, test_gpu$Core_Speed)

cat("GPU Linear Regression Performance (R):\n")
print(lm_perf_gpu)

png("outputs/r_gpu_pred_vs_actual.png", width = 800, height = 600, res = 120)
plot(test_gpu$Core_Speed, lm_preds_gpu,
     col = "#27ae60", pch = 19,
     xlab = "Actual GPU Core Speed (MHz)",
     ylab = "Predicted Core Speed (MHz)",
     main = "R Linear Regression: GPU Predicted vs Actual")
abline(a = 0, b = 1, col = "#d9534f", lwd = 2, lty = 2)
dev.off()

cat("\nR Baseline Workflow Executed Successfully.\n")
cat("======================================================================\n")
