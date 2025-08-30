library(CASdatasets)
library(MASS)  # for fitdistr

data(freMTPL2freq)
data(freMTPL2sev)

# Aggregate claim amount per policy
agg_sev <- aggregate(ClaimAmount ~ IDpol, data = freMTPL2sev, sum)
df <- merge(freMTPL2freq, agg_sev, by = "IDpol", all.x = TRUE)
df$ClaimAmount[is.na(df$ClaimAmount)] <- 0
df <- subset(df, Exposure > 0)
df <- df[, c("IDpol", "Exposure", "ClaimNb", "ClaimAmount")]
set.seed(42)
n <- nrow(df)
train_idx <- sample(seq_len(n), size = 0.8 * n)
df_train <- df[train_idx, ]
df_test  <- df[-train_idx, ]

lambda_hat <- sum(df_train$ClaimNb) / sum(df_train$Exposure)
cat("Estimated lambda:", lambda_hat, "\n")


# Get all claim severities (per claim)
claim_sev_train <- freMTPL2sev[freMTPL2sev$IDpol %in% df_train$IDpol, "ClaimAmount"]

# Use per-claim severity data from training set
claim_sev_train <- freMTPL2sev$ClaimAmount[freMTPL2sev$IDpol %in% df_train$IDpol]
claim_sev_train <- claim_sev_train[claim_sev_train > 0]

# Sample moments
mean_sev <- mean(claim_sev_train)
var_sev  <- var(claim_sev_train)

# Moment estimators
alpha_hat <- mean_sev^2 / var_sev
beta_hat  <- var_sev / mean_sev  # This is the scale (not rate)

mu_hat <- alpha_hat * beta_hat  # Should equal mean_sev

cat("Gamma shape (α):", round(alpha_hat, 4), "\n")
cat("Gamma scale (β):", round(beta_hat, 4), "\n")


set.seed(10)
M <- 1000  # number of simulated samples of total loss

# Store total aggregate loss for each simulation
sim_total_loss <- numeric(M)

for (m in 1:M) {
  S_j <- numeric(nrow(df_test))
  for (j in 1:nrow(df_test)) {
    # Simulate claim count
    n_j <- rpois(1, lambda = lambda_hat * df_test$Exposure[j])
    
    # Simulate aggregate claim
    if (n_j > 0) {
      S_j[j] <- sum(rgamma(n_j, shape = alpha_hat, scale = beta_hat))
    } else {
      S_j[j] <- 0
    }
  }
  sim_total_loss[m] <- sum(S_j)
}


actual_loss <- sum(df_test$ClaimAmount)

hist(sim_total_loss, breaks = 40, main = "Predictive Distribution vs Actual Aggregate Loss",
     xlab = "Aggregate Loss (Validation Set)", col = "lightblue")
abline(v = actual_loss, col = "red", lwd = 2)
legend("topright", legend = c("Actual Aggregate Loss"), col = c("red"), lwd = 2)

cat("Actual aggregate loss:", actual_loss, "\n")
cat("Mean simulated loss:", mean(sim_total_loss), "\n")
cat("95% CI of simulated loss:", quantile(sim_total_loss, c(0.025, 0.975)), "\n")


