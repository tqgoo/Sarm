# --- Libraries ---
library(keras)
library(tensorflow)
library(reticulate)
library(MASS)
library(knitr)
library(dplyr)
library(tibble)
library(moments)
np <- import("numpy")

# ---Pure in-sample comparison ----
run_all_models_and_compare <- function(
    X, freq, sev_list
) {
  # --- Helper: Transform predicted to observable mean ---
  get_policy_predictions <- function(fit, ages, dist, sevtype) {
    # Frequency
    x_scaled <- (ages - fit$scaler$age_mean) / fit$scaler$age_sd
    x_pred <- np$array(matrix(as.numeric(x_scaled), ncol=1), dtype="float32")
    
    if (dist == "poisson") {
      freq_pred_scaled <- as.vector(fit$poisson_model$predict(x_pred))
      freq_pred <- freq_pred_scaled * fit$scaler$count_sd + fit$scaler$count_mean
    } else if (dist == "nb") {
      p_scaled <- as.vector(fit$nb_model$predict(x_pred))
      p_pred <- p_scaled * fit$scaler$p_sd + fit$scaler$p_mean
      p_pred <- pmin(pmax(p_pred, 1e-8), 1-1e-8)
      freq_pred <- fit$r_hat * (1 - p_pred) / p_pred
    } else if (dist == "zip") {
      # If NN outputs only scaled lambda, and pi_hat is a constant:
      freq_pred_scaled <- as.vector(fit$zip_model$predict(x_pred))
      lambda_pred <- freq_pred_scaled * fit$scaler$freq_sd + fit$scaler$freq_mean
      pi_hat <- if (!is.null(fit$pi_hat)) fit$pi_hat else 0
      freq_pred <- (1 - pi_hat) * lambda_pred
      # If NN outputs both logit_pi and lambda, use:
      # zip_pred <- fit$zip_model$predict(x_pred) # matrix, 2 cols
      # pi_hat <- 1 / (1 + exp(-zip_pred[,1]))
      # lambda_pred <- zip_pred[,2] * fit$scaler$freq_sd + fit$scaler$freq_mean
      # freq_pred <- (1 - pi_hat) * lambda_pred
    } else {
      stop("Unknown frequency distribution")
    }
    
    # Severity
    age_pol_scaled <- (ages - fit$scaler$age_pol_mean) / fit$scaler$age_pol_sd
    xg_pred <- np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")
    if (sevtype == "gamma") {
      beta_pred_scaled <- as.vector(fit$gamma_model$predict(xg_pred))
      beta_pred <- beta_pred_scaled * fit$scaler$sev_mean_sd + fit$scaler$sev_mean_mean
      # mean severity per claim: alpha * beta
      sev_pred <- fit$alpha_hat * beta_pred
    } else if (sevtype == "lognormal") {
      mu_pred_scaled <- as.vector(fit$ln_model$predict(xg_pred))
      mu_pred <- mu_pred_scaled * fit$scaler$mu_implied_sd + fit$scaler$mu_implied_mean
      # mean severity: exp(mu + 0.5*sigma^2)
      sev_pred <- exp(mu_pred + 0.5 * fit$sigma_hat^2)
    } else {
      stop("Unknown severity distribution")
    }
    list(freq_pred = freq_pred, sev_pred = sev_pred)
  }
  
  # --- RMSE on actual values (not standardized) ---
  rmse <- function(est, true) {
    idx <- which(!is.na(est) & !is.na(true) & !is.nan(est) & !is.nan(true))
    if (length(idx) == 0) return(NA_real_)
    sqrt(mean((est[idx] - true[idx])^2))
  }
  
  # --- Fit models ---
  cat("Fitting Poisson-Gamma...\n")
  fit_pg <- fit_poisson_gamma(X, freq, sev_list, kernel = "exponential")
  cat("Fitting ZIP-Gamma...\n")
  fit_zg <- fit_zip_gamma(X, freq, sev_list, kernel = "exponential")
  cat("Fitting NB-Gamma...\n")
  fit_ng <- fit_nb_gamma(X, freq, sev_list, kernel = "exponential")
  cat("Fitting Poisson-Lognormal...\n")
  fit_pl <- fit_poisson_lognormal(X, freq, sev_list, kernel = "exponential")
  cat("Fitting ZIP-Lognormal...\n")
  fit_zl <- fit_zip_lognormal(X, freq, sev_list, kernel = "exponential")
  cat("Fitting NB-Lognormal...\n")
  fit_nl <- fit_nb_lognormal(X, freq, sev_list, kernel = "exponential")
  
  fits <- list(
    poisson_gamma = fit_pg,
    zip_gamma = fit_zg,
    nb_gamma = fit_ng,
    poisson_lognormal = fit_pl,
    zip_lognormal = fit_zl,
    nb_lognormal = fit_nl
  )
  model_names <- c("Poisson-Gamma", "ZIP-Gamma", "NB-Gamma",
                   "Poisson-Lognormal", "ZIP-Lognormal", "NB-Lognormal")
  dists   <- c("poisson", "zip", "nb", "poisson", "zip", "nb")
  sevs    <- c("gamma", "gamma", "gamma", "lognormal", "lognormal", "lognormal")
  
  # --- Prepare observed values for RMSE on dataset ---
  obs_freq     <- freq
  obs_sev_mean <- sapply(sev_list, function(x) if(length(x) > 0) mean(x) else NA_real_)
  obs_agg      <- sapply(sev_list, sum)
  
  # --- Compute all model predictions for observed policies ---
  rmse_freq <- numeric(6)
  rmse_sev  <- numeric(6)
  rmse_agg  <- numeric(6)
  for (j in seq_along(model_names)) {
    pred <- get_policy_predictions(fits[[j]], X[,1], dists[j], sevs[j])
    # RMSE Frequency: all policies
    rmse_freq[j] <- rmse(pred$freq_pred, obs_freq)
    # RMSE Severity: only policies with nonzero claims
    idx_nonzero <- which(obs_freq > 0 & !is.na(obs_sev_mean))
    rmse_sev[j] <- rmse(pred$sev_pred[idx_nonzero], obs_sev_mean[idx_nonzero])
    # RMSE Aggregate: sum(predicted freq * predicted mean severity)
    pred_agg <- pred$freq_pred * pred$sev_pred
    rmse_agg[j] <- rmse(pred_agg, obs_agg)
  }
  
  # --- Assemble table ---
  stats <- tibble::tibble(
    Model          = model_names,
    Omega          = c(fit_pg$omega_nn, fit_zg$omega_nn, fit_ng$omega_nn,
                       fit_pl$omega_nn, fit_zl$omega_nn, fit_nl$omega_nn),
    RMSE_Frequency = rmse_freq,
    RMSE_Severity  = rmse_sev,
    RMSE_Aggregate = rmse_agg
  )
  
  # Optionally also return predictions, fits, etc.
  return(list(
    stats_table = stats,
    fits = fits
  ))
}




# Run:
results <- run_all_models_and_compare(X, freq, sev_list)

# Show the comparison table
print(results$stats_table)

evaluate_models_on_validation <- function(
    fits, X_val, freq_val, sev_list_val
) {
  require(tibble)
  np <- reticulate::import("numpy")
  
  n_val <- nrow(X_val)
  model_names <- c("Poisson-Gamma", "ZIP-Gamma", "NB-Gamma",
                   "Poisson-Lognormal", "ZIP-Lognormal", "NB-Lognormal")
  
  # RMSE helper
  rmse <- function(est, true) {
    idx <- which(!is.na(est) & !is.na(true) & !is.nan(est) & !is.nan(true))
    if (length(idx) == 0) return(NA_real_)
    sqrt(mean((est[idx] - true[idx])^2))
  }
  
  # Predict observable means: frequency and severity
  get_policy_predictions <- function(fit, ages, dist, sevtype) {
    # Frequency
    x_scaled <- (ages - fit$scaler$age_mean) / fit$scaler$age_sd
    x_pred <- np$array(matrix(as.numeric(x_scaled), ncol=1), dtype="float32")
    if (dist == "poisson") {
      freq_pred_scaled <- as.vector(fit$poisson_model$predict(x_pred))
      freq_pred <- freq_pred_scaled * fit$scaler$count_sd + fit$scaler$count_mean
    } else if (dist == "nb") {
      p_scaled <- as.vector(fit$nb_model$predict(x_pred))
      p_pred <- p_scaled * fit$scaler$p_sd + fit$scaler$p_mean
      p_pred <- pmin(pmax(p_pred, 1e-8), 1-1e-8)
      freq_pred <- fit$r_hat * (1 - p_pred) / p_pred
    } else if (dist == "zip") {
      freq_pred_scaled <- as.vector(fit$zip_model$predict(x_pred))
      lambda_pred <- freq_pred_scaled * fit$scaler$freq_sd + fit$scaler$freq_mean
      pi_hat <- if (!is.null(fit$pi_hat)) fit$pi_hat else 0
      freq_pred <- (1 - pi_hat) * lambda_pred
    } else {
      stop("Unknown frequency distribution")
    }
    
    # Severity
    age_pol_scaled <- (ages - fit$scaler$age_pol_mean) / fit$scaler$age_pol_sd
    xg_pred <- np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")
    if (sevtype == "gamma") {
      beta_pred_scaled <- as.vector(fit$gamma_model$predict(xg_pred))
      beta_pred <- beta_pred_scaled * fit$scaler$sev_mean_sd + fit$scaler$sev_mean_mean
      sev_pred <- fit$alpha_hat * beta_pred # mean per claim
    } else if (sevtype == "lognormal") {
      mu_pred_scaled <- as.vector(fit$ln_model$predict(xg_pred))
      mu_pred <- mu_pred_scaled * fit$scaler$mu_implied_sd + fit$scaler$mu_implied_mean
      sev_pred <- exp(mu_pred + 0.5 * fit$sigma_hat^2)
    } else {
      stop("Unknown severity distribution")
    }
    list(freq_pred = freq_pred, sev_pred = sev_pred)
  }
  
  # Prepare observed values
  obs_freq     <- freq_val
  obs_sev_mean <- sapply(sev_list_val, function(x) if(length(x) > 0) mean(x) else NA_real_)
  obs_agg      <- sapply(sev_list_val, sum)
  
  # Model definitions
  dists <- c("poisson", "zip", "nb", "poisson", "zip", "nb")
  sevs  <- c("gamma", "gamma", "gamma", "lognormal", "lognormal", "lognormal")
  
  # Loop and compute RMSEs
  rmse_freq <- numeric(6)
  rmse_sev  <- numeric(6)
  rmse_agg  <- numeric(6)
  for (j in seq_along(model_names)) {
    pred <- get_policy_predictions(fits[[j]], X_val[,1], dists[j], sevs[j])
    # RMSE Frequency: all policies
    rmse_freq[j] <- rmse(pred$freq_pred, obs_freq)
    # RMSE Severity: only policies with nonzero claims
    idx_nonzero <- which(obs_freq > 0 & !is.na(obs_sev_mean))
    rmse_sev[j] <- rmse(pred$sev_pred[idx_nonzero], obs_sev_mean[idx_nonzero])
    # RMSE Aggregate: sum(predicted freq * predicted mean severity)
    pred_agg <- pred$freq_pred * pred$sev_pred
    rmse_agg[j] <- rmse(pred_agg, obs_agg)
  }
  
  # Assemble output table
  stats <- tibble::tibble(
    Model          = model_names,
    Omega          = c(fits$poisson_gamma$omega_nn, fits$zip_gamma$omega_nn, fits$nb_gamma$omega_nn,
                       fits$poisson_lognormal$omega_nn, fits$zip_lognormal$omega_nn, fits$nb_lognormal$omega_nn),
    RMSE_Frequency = rmse_freq,
    RMSE_Severity  = rmse_sev,
    RMSE_Aggregate = rmse_agg
  )
  return(stats)
}



# Validation data
# X_val: matrix/dataframe, freq_val: vector, sev_list_val: list of vectors
# Suppose you have the following splits
# (example: use 8000 for train, 2000 for validation)
n_train <- 8000
n_val <- 2000

X_train <- X[1:n_train, , drop=FALSE]
freq_train <- freq[1:n_train]
sev_list_train <- sev_list[1:n_train]

X_val <- X[(n_train+1):(n_train+n_val), , drop=FALSE]
freq_val <- freq[(n_train+1):(n_train+n_val)]
sev_list_val <- sev_list[(n_train+1):(n_train+n_val)]


# Assuming you already split X, freq, sev_list as:
# X_train, freq_train, sev_list_train (for training)
# X_val, freq_val, sev_list_val       (for validation)

results_nn <- run_all_models_and_compare(X_train, freq_train, sev_list_train)

stats_val <- evaluate_models_on_validation(
  fits = results_nn$fits,
  X_val = X_val,
  freq_val = freq_val,
  sev_list_val = sev_list_val
)

print(stats_val)













