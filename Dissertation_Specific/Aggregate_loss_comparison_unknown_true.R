compare_aggregate_loss_bootstrap <- function(
    all_models, X, freq, sev_list, 
    n_boot = 5000,  # number of bootstrap replicates
    summary_probs = c(0.05, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999),
    seed = 369
) {
  requireNamespace("e1071")
  set.seed(seed)
  M <- nrow(X)
  B <- n_boot
  model_names <- c("Poisson-Gamma", "ZIP-Gamma", "NB-Gamma", 
                   "Poisson-Lognormal", "ZIP-Lognormal", "NB-Lognormal")
  fits <- all_models$fits
  
  # Frequency prediction helper
  predict_freq <- function(model, xmat, dist) {
    if (dist == "poisson") {
      x_scaled <- (xmat - model$scaler$age_mean) / model$scaler$age_sd
      x_pred <- np$array(matrix(as.numeric(x_scaled), ncol=1), dtype="float32")
      pred_scaled <- as.vector(model$poisson_model$predict(x_pred))
      pred <- pred_scaled * model$scaler$count_sd + model$scaler$count_mean
      return(pred)
    } else if (dist == "zip") {
      x_scaled <- (xmat - model$scaler$age_mean) / model$scaler$age_sd
      x_pred <- np$array(matrix(as.numeric(x_scaled), ncol=1), dtype="float32")
      pred_scaled <- as.vector(model$zip_model$predict(x_pred))
      pred <- pred_scaled * model$scaler$freq_sd + model$scaler$freq_mean
      return(pred)
    } else if (dist == "nb") {
      x_scaled <- (xmat - model$scaler$age_mean) / model$scaler$age_sd
      x_pred <- np$array(matrix(as.numeric(x_scaled), ncol=1), dtype="float32")
      p_scaled <- as.vector(model$nb_model$predict(x_pred))
      p_pred <- p_scaled * model$scaler$p_sd + model$scaler$p_mean
      p_pred <- pmin(pmax(p_pred, 1e-8), 1-1e-8)
      mu_pred <- model$r_hat * (1 - p_pred) / p_pred
      return(mu_pred)
    }
    stop("Unknown distribution.")
  }
  
  # Severity prediction helper
  predict_sev <- function(model, xmat, sevtype) {
    if (sevtype == "gamma") {
      x_scaled <- (xmat - model$scaler$age_pol_mean) / model$scaler$age_pol_sd
      x_pred <- np$array(matrix(as.numeric(x_scaled), ncol=1), dtype="float32")
      beta_scaled <- as.vector(model$gamma_model$predict(x_pred))
      beta_pred <- beta_scaled * model$scaler$sev_mean_sd + model$scaler$sev_mean_mean
      return(beta_pred)
    } else if (sevtype == "lognormal") {
      x_scaled <- (xmat - model$scaler$age_pol_mean) / model$scaler$age_pol_sd
      x_pred <- np$array(matrix(as.numeric(x_scaled), ncol=1), dtype="float32")
      mu_scaled <- as.vector(model$ln_model$predict(x_pred))
      mu_pred <- mu_scaled * model$scaler$mu_implied_sd + model$scaler$mu_implied_mean
      return(mu_pred)
    }
    stop("Unknown severity type.")
  }
  
  # Prepare for storing bootstrapped aggregate losses for models
  agg_mat <- matrix(NA, nrow=B, ncol=6)
  colnames(agg_mat) <- model_names
  
  # Prepare for storing "Truth"
  agg_loss_true <- numeric(B)
  
  for (b in 1:B) {
    idx <- sample(1:M, M, replace=TRUE)
    ages_boot <- X[idx, 1]
    freq_boot <- freq[idx]
    sev_boot  <- sev_list[idx]
    # Truth aggregate loss
    agg_loss_true[b] <- sum(unlist(sev_boot))
    
    # For each model: simulate aggregate loss
    for (j in 1:6) {
      # model/family pairing
      if      (j == 1) { fit <- fits$poisson_gamma;    dist <- "poisson"; sevtype <- "gamma"    }
      else if (j == 2) { fit <- fits$zip_gamma;        dist <- "zip";     sevtype <- "gamma"    }
      else if (j == 3) { fit <- fits$nb_gamma;         dist <- "nb";      sevtype <- "gamma"    }
      else if (j == 4) { fit <- fits$poisson_lognormal;dist <- "poisson"; sevtype <- "lognormal"}
      else if (j == 5) { fit <- fits$zip_lognormal;    dist <- "zip";     sevtype <- "lognormal"}
      else if (j == 6) { fit <- fits$nb_lognormal;     dist <- "nb";      sevtype <- "lognormal"}
      
      # Predict frequency and severity params for each policy in bootstrap
      freq_pred <- predict_freq(fit, ages_boot, dist)
      sev_pred  <- predict_sev(fit, ages_boot, sevtype)
      
      # Simulate aggregate loss for this bootstrap sample
      agg_loss <- 0
      for (k in 1:length(ages_boot)) {
        n_claims <- 0
        if (dist == "poisson") {
          n_claims <- rpois(1, freq_pred[k])
        } else if (dist == "zip") {
          pi_hat <- if (!is.null(fit$pi_hat)) fit$pi_hat else 0
          is_zero <- rbinom(1, 1, pi_hat)
          n_claims <- if (is_zero) 0 else rpois(1, freq_pred[k])
        } else if (dist == "nb") {
          r_hat <- fit$r_hat
          mu_nb <- freq_pred[k]
          n_claims <- rnbinom(1, size=r_hat, mu=mu_nb)
        }
        # Draw severities
        if (n_claims > 0) {
          if (sevtype == "gamma") {
            alpha_hat <- fit$alpha_hat
            beta_hat  <- sev_pred[k]
            agg_loss <- agg_loss + sum(rgamma(n_claims, shape=alpha_hat, scale=beta_hat))
          } else if (sevtype == "lognormal") {
            mu_hat    <- sev_pred[k]
            sigma_hat <- fit$sigma_hat
            agg_loss <- agg_loss + sum(rlnorm(n_claims, meanlog=mu_hat, sdlog=sigma_hat))
          }
        }
      }
      agg_mat[b, j] <- agg_loss
    }
    if (b %% 500 == 0) cat("Bootstrap sample", b, "of", B, "\n")
  }
  
  # Summarize statistics for each model + truth
  agg_stats <- as.data.frame(matrix(NA, nrow=7, ncol=12))
  colnames(agg_stats) <- c("Model", "Mean", "SD", "Skewness", "Kurtosis", 
                           paste0(round(summary_probs*100, 1), "%"))
  # Truth
  vals <- agg_loss_true
  agg_stats[1, 1] <- "Truth"
  agg_stats[1, 2] <- mean(vals, na.rm=TRUE)
  agg_stats[1, 3] <- sd(vals, na.rm=TRUE)
  agg_stats[1, 4] <- if (requireNamespace("e1071", quietly=TRUE)) e1071::skewness(vals, na.rm=TRUE) else NA
  agg_stats[1, 5] <- if (requireNamespace("e1071", quietly=TRUE)) e1071::kurtosis(vals, na.rm=TRUE) else NA
  agg_stats[1, 6:12] <- as.numeric(quantile(vals, probs=summary_probs, na.rm=TRUE))
  
  # Models
  for (j in 1:6) {
    vals <- agg_mat[, j]
    agg_stats[j+1, 1] <- model_names[j]
    agg_stats[j+1, 2] <- mean(vals, na.rm=TRUE)
    agg_stats[j+1, 3] <- sd(vals, na.rm=TRUE)
    agg_stats[j+1, 4] <- if (requireNamespace("e1071", quietly=TRUE)) e1071::skewness(vals, na.rm=TRUE) else NA
    agg_stats[j+1, 5] <- if (requireNamespace("e1071", quietly=TRUE)) e1071::kurtosis(vals, na.rm=TRUE) else NA
    agg_stats[j+1, 6:12] <- as.numeric(quantile(vals, probs=summary_probs, na.rm=TRUE))
  }
  rownames(agg_stats) <- NULL
  return(agg_stats)
}



# all_models: result from run_all_models_and_compare(...)
# X, freq, sev_list: your original data
agg_stats <- compare_aggregate_loss_bootstrap(results, X, freq, sev_list, n_boot=50)
print(agg_stats)
