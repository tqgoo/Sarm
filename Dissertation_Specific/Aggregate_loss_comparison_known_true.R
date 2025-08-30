set.seed(888)
M_val <- 10000
alpha0 <- 2.87
omega0 <- 0.3

age_val <- runif(M_val, 20, 60)
lambda_true_val <- 0.003*(age_val-30)^2 + exp(0.05*(age_val-40) - 0.008*(age_val-40)^2 + log(3)) + 0.6
mu_true_val     <- 0.08*abs(age_val-40) + exp(0.3*sqrt(age_val) - 0.12*(age_val-40)^2 + log(0.5)) + 2
beta_true_val   <- mu_true_val / alpha0

# For each policy, simulate frequency and severities
freq_val <- integer(M_val)
sev_list_val <- vector("list", M_val)
for(i in seq_len(M_val)) {
  lam <- lambda_true_val[i]
  n_i <- rpois(1, lam)
  freq_val[i] <- n_i
  if(n_i > 0L) {
    psi_i <- (n_i - lam)/sqrt(lam)
    draw_one <- function() {
      repeat {
        x_prop <- rgamma(1, shape=alpha0, scale=beta_true_val[i])
        w      <- 1 + omega0 * psi_i * (x_prop - mu_true_val[i])/(sqrt(alpha0)*beta_true_val[i])
        if (w>0 && runif(1) < w) return(x_prop)
      }
    }
    sev_list_val[[i]] <- replicate(n_i, draw_one())
  } else {
    sev_list_val[[i]] <- numeric(0)
  }
}


X_val <- matrix(age_val, ncol = 1)

agg_loss_true <- sapply(sev_list_val, sum)
total_loss_true <- sum(agg_loss_true)


library(moments)

simulate_aggregate_loss <- function(fit, X_val, M_val, model_type = c("poisson", "zip", "nb"),
                                    sev_type = c("gamma", "lognormal"), pi_hat = NULL) {
  age_val <- as.numeric(X_val[, 1])
  
  # --- Predict Frequency ---
  if (model_type == "poisson") {
    x_pred <- np$array(matrix((age_val - fit$scaler$age_mean) / fit$scaler$age_sd, ncol=1), dtype="float32")
    lambda_pred_scaled <- as.vector(fit$poisson_model$predict(x_pred))
    lambda_pred <- lambda_pred_scaled * fit$scaler$count_sd + fit$scaler$count_mean
    freq_pred <- rpois(M_val, lambda_pred)
  } else if (model_type == "zip") {
    x_pred <- np$array(matrix((age_val - fit$scaler$age_mean) / fit$scaler$age_sd, ncol=1), dtype="float32")
    lambda_pred_scaled <- as.vector(fit$zip_model$predict(x_pred))
    lambda_pred <- lambda_pred_scaled * fit$scaler$freq_sd + fit$scaler$freq_mean
    # Use global pi_hat
    pi_hat <- fit$pi_hat
    freq_pred <- rbinom(M_val, 1, 1-pi_hat) * rpois(M_val, lambda_pred)
  } else if (model_type == "nb") {
    x_pred <- np$array(matrix((age_val - fit$scaler$age_mean) / fit$scaler$age_sd, ncol=1), dtype="float32")
    p_pred_scaled <- as.vector(fit$nb_model$predict(x_pred))
    p_pred <- p_pred_scaled * fit$scaler$p_sd + fit$scaler$p_mean
    p_pred <- pmin(pmax(p_pred, 1e-8), 1-1e-8)
    r_hat <- fit$r_hat
    freq_pred <- rnbinom(M_val, size = r_hat, prob = p_pred)
  } else {
    stop("Unknown model_type")
  }
  
  # --- Predict Severity ---
  age_pol_idx <- which(freq_pred > 0)
  age_val_pol <- age_val[age_pol_idx]
  n_pol <- freq_pred[age_pol_idx]
  
  if (sev_type == "gamma") {
    xg_pred <- np$array(matrix((age_val_pol - fit$scaler$age_pol_mean) / fit$scaler$age_pol_sd, ncol=1), dtype="float32")
    beta_pred_scaled <- as.vector(fit$gamma_model$predict(xg_pred))
    beta_pred <- beta_pred_scaled * fit$scaler$sev_mean_sd + fit$scaler$sev_mean_mean
    alpha_hat <- fit$alpha_hat
    sev_draws <- lapply(seq_along(age_pol_idx), function(j) rgamma(n_pol[j], shape=alpha_hat, scale=beta_pred[j]))
  } else if (sev_type == "lognormal") {
    xg_pred <- np$array(matrix((age_val_pol - fit$scaler$age_pol_mean) / fit$scaler$age_pol_sd, ncol=1), dtype="float32")
    mu_pred_scaled <- as.vector(fit$ln_model$predict(xg_pred))
    mu_pred <- mu_pred_scaled * fit$scaler$mu_implied_sd + fit$scaler$mu_implied_mean
    sigma_hat <- fit$sigma_hat
    sev_draws <- lapply(seq_along(age_pol_idx), function(j) rlnorm(n_pol[j], meanlog=mu_pred[j], sdlog=sigma_hat))
  } else {
    stop("Unknown sev_type")
  }
  
  # Combine all losses into portfolio total
  agg_loss <- numeric(M_val)
  agg_loss[] <- 0
  for(j in seq_along(age_pol_idx)) {
    i <- age_pol_idx[j]
    agg_loss[i] <- sum(sev_draws[[j]])
  }
  agg_loss
}

get_loss_stats <- function(loss_vec) {
  q <- quantile(loss_vec, probs = c(0.05, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999))
  c(
    Mean = mean(loss_vec),
    Std  = sd(loss_vec),
    Skew = skewness(loss_vec),
    Kurtosis = kurtosis(loss_vec),
    `5%` = q[1],
    `50%` = q[2],
    `75%` = q[3],
    `90%` = q[4],
    `95%` = q[5],
    `99%` = q[6],
    `99.9%` = q[7]
  )
}


fits_nn <- results_nn$fits
# Truth
agg_loss_true <- sapply(sev_list_val, sum)
stats_true <- get_loss_stats(agg_loss_true)

# Each fitted model
agg_pg_nn <- simulate_aggregate_loss(fits_nn$poisson_gamma, X_val, M_val, "poisson", "gamma")
agg_zg_nn <- simulate_aggregate_loss(fits_nn$zip_gamma,     X_val, M_val, "zip",     "gamma")
agg_ng_nn <- simulate_aggregate_loss(fits_nn$nb_gamma,      X_val, M_val, "nb",      "gamma")
agg_pl_nn <- simulate_aggregate_loss(fits_nn$poisson_lognormal, X_val, M_val, "poisson", "lognormal")
agg_zl_nn <- simulate_aggregate_loss(fits_nn$zip_lognormal,     X_val, M_val, "zip",     "lognormal")
agg_nl_nn <- simulate_aggregate_loss(fits_nn$nb_lognormal,      X_val, M_val, "nb",      "lognormal")

stats_pg_nn <- get_loss_stats(agg_pg_nn)
stats_zg_nn <- get_loss_stats(agg_zg_nn)
stats_ng_nn <- get_loss_stats(agg_ng_nn)
stats_pl_nn <- get_loss_stats(agg_pl_nn)
stats_zl_nn <- get_loss_stats(agg_zl_nn)
stats_nl_nn <- get_loss_stats(agg_nl_nn)

# Combine into one table
library(tibble)
agg_stats <- rbind(
  Truth             = stats_true,
  `Poisson-Gamma`   = stats_pg_nn,
  `ZIP-Gamma`       = stats_zg_nn,
  `NB-Gamma`        = stats_ng_nn,
  `Poisson-Lognormal` = stats_pl_nn,
  `ZIP-Lognormal`     = stats_zl_nn,
  `NB-Lognormal`      = stats_nl_nn
)
print(round(agg_stats, 3))
