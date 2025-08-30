# Load the data (uncomment if needed)
library(CASdatasets)
data(ausprivauto0405)
df <- ausprivauto0405

# Sanity check: print names
print(names(df))

# Step 1: Select useful columns
# You may adjust variables as desired!
cat_vars <- c("VehAge", "DrivAge", "VehBody", "Gender")
num_vars <- c("VehValue", "Exposure")

# Step 2: One-hot encode categorical variables
X_cat <- model.matrix(~ VehAge + DrivAge + VehBody + Gender, data=df)[, -1]

# Step 3: Combine with numeric variables
X <- cbind(
  X_cat,
  VehValue = df$VehValue,
  Exposure = df$Exposure
)
X <- as.matrix(X)

# Step 4: Frequency vector
freq <- df$ClaimNb

# Step 5: Severity list (list of per-claim severities for each policy)
sev_list <- vector("list", nrow(df))
for (i in seq_len(nrow(df))) {
  n_claims <- df$ClaimNb[i]
  total_amt <- df$ClaimAmount[i]
  if (!is.na(n_claims) && n_claims > 0) {
    sev_list[[i]] <- rep(total_amt / n_claims, n_claims)
  } else {
    sev_list[[i]] <- numeric(0)
  }
}

# Sanity check: dimensions
cat(sprintf("X: %d rows, %d columns\n", nrow(X), ncol(X)))
cat(sprintf("freq: %d elements\n", length(freq)))
cat(sprintf("sev_list: %d policies, first 3: %s\n",
            length(sev_list),
            paste(sapply(sev_list[1:3], length), collapse=", "))
)


set.seed(42)  # for reproducibility

N <- nrow(X)
idx <- sample(seq_len(N))         # shuffle row indices
n_train <- floor(0.9 * N)         # 90% for training

train_idx <- idx[1:n_train]
val_idx   <- idx[(n_train + 1):N]

# Split covariates
X_train <- X[train_idx, , drop=FALSE]
X_val   <- X[val_idx, , drop=FALSE]

# Split frequency vector
freq_train <- freq[train_idx]
freq_val   <- freq[val_idx]

# Split severity list
sev_list_train <- sev_list[train_idx]
sev_list_val   <- sev_list[val_idx]

# (Optional) Check sizes
cat("Training set size:", length(train_idx), "\n")
cat("Validation set size:", length(val_idx), "\n")

