library(ggplot2)

plot_sarmanov_nn_fit <- function(
    fit,
    X, freq, sev_list,
    train_idx, val_idx,
    covariate_name = NULL,
    Msim = 50000 # Number of simulations for aggregate
) {
  # Extract scaling info
  X <- as.matrix(X)
  covariate <- if (is.null(covariate_name)) X[,1] else X[,covariate_name]
  # For plot labels
  freq_dist <- fit$freq_dist
  sev_dist <- fit$sev_dist
  scaler <- fit$scaler
  freq_model <- fit$freq_model
  sev_model <- fit$sev_model
  pi_hat <- fit$pi_hat
  r_hat <- fit$r_hat
  alpha_hat <- fit$alpha_hat
  sigma_hat <- fit$sigma_hat
  hyper <- fit$hyper
  p_mean <- scaler$p_mean
  p_sd <- scaler$p_sd
  freq_mean <- scaler$freq_mean
  freq_sd <- scaler$freq_sd
  sev_mean_mean <- scaler$sev_mean_mean
  sev_mean_sd <- scaler$sev_mean_sd
  X_mean <- scaler$X_mean
  X_sd <- scaler$X_sd
  
  # Helper to predict (returns all implied parameters for any index)
  predict_params <- function(idx) {
    X_ <- X[idx,,drop=FALSE]
    covariate_ <- covariate[idx]
    X_scaled_ <- scale(X_, center = X_mean, scale = X_sd)
    freq_pred_scaled <- as.vector(freq_model$predict(
      np$array(matrix(as.numeric(X_scaled_), ncol=ncol(X)), dtype="float32")))
    # Freq
    if (freq_dist == "Poisson") {
      lambda_pred <- freq_pred_scaled * freq_sd + freq_mean
      lambda_true <- freq[idx]
      list(freq_par_est = lambda_pred,
           freq_par_true = lambda_true)
    } else if (freq_dist == "ZIP") {
      lambda_pred <- freq_pred_scaled * freq_sd + freq_mean
      lambda_est <- (1 - pi_hat) * lambda_pred
      lambda_true <- freq[idx]
      list(freq_par_est = lambda_est,
           freq_par_true = lambda_true)
    } else if (freq_dist == "NB") {
      p_pred <- freq_pred_scaled * p_sd + p_mean
      p_pred <- pmin(pmax(p_pred, 1e-8), 1-1e-8)
      mu_est <- r_hat * (1 - p_pred) / p_pred
      mu_true <- freq[idx]
      list(freq_par_est = mu_est,
           freq_par_true = mu_true)
    }
  }
  
  # Helper for severity
  predict_sev_param <- function(idx) {
    X_ <- X[idx,,drop=FALSE]
    sev_idx <- which(freq[idx] > 0)
    if (length(sev_idx) == 0) return(NULL)
    X_pol_ <- X_[sev_idx,,drop=FALSE]
    X_pol_scaled <- scale(X_pol_, center = scaler$X_pol_mean, scale = scaler$X_pol_sd)
    sev_pred_scaled <- as.vector(sev_model$predict(
      np$array(matrix(as.numeric(X_pol_scaled), ncol=ncol(X)), dtype="float32")))
    if (sev_dist == "Gamma") {
      beta_est <- sev_pred_scaled * sev_mean_sd + sev_mean_mean
      beta_true <- sapply(sev_list[idx[sev_idx]], mean)
      list(sev_par_est = beta_est, sev_par_true = beta_true, idx = idx[sev_idx])
    } else if (sev_dist == "Lognormal") {
      mu_log_est <- sev_pred_scaled * sev_mean_sd + sev_mean_mean
      mu_log_true <- log(sapply(sev_list[idx[sev_idx]], mean))
      list(sev_par_est = mu_log_est, sev_par_true = mu_log_true, idx = idx[sev_idx])
    }
  }
  
  ## --- Plotting frequency
  plot_freq <- function(idx, label) {
    par <- predict_params(idx)
    df <- data.frame(
      covariate = covariate[idx],
      est = par$freq_par_est,
      true = par$freq_par_true
    )
    p <- ggplot(df, aes(x = covariate)) +
      geom_point(aes(y = true), color="black", alpha=0.2, size=1, shape=16) +
      geom_line(aes(y = est), color="red", lwd=1.2) +
      labs(title = paste(label, "- Frequency Parameter"), y = "Parameter (mean of freq)", x = covariate_name %||% "Covariate") +
      theme_minimal()
    print(p)
  }
  
  ## --- Plotting severity
  plot_sev <- function(idx, label) {
    sev_out <- predict_sev_param(idx)
    if (is.null(sev_out)) {
      cat("No positive freq policies in", label, "\n"); return(NULL)
    }
    df <- data.frame(
      covariate = covariate[sev_out$idx],
      est = sev_out$sev_par_est,
      true = sev_out$sev_par_true
    )
    p <- ggplot(df, aes(x = covariate)) +
      geom_point(aes(y = true), color="black", alpha=0.2, size=1, shape=16) +
      geom_line(aes(y = est), color="blue", lwd=1.2) +
      labs(title = paste(label, "- Severity Parameter"), y = "Parameter (mean of sev)", x = covariate_name %||% "Covariate") +
      theme_minimal()
    print(p)
  }
  
  ## --- Simulate aggregate losses for validation set
  simulate_aggregate <- function(idx) {
    X_ <- X[idx,,drop=FALSE]
    freq_pred_scaled <- as.vector(freq_model$predict(
      np$array(matrix(as.numeric(scale(X_, center = X_mean, scale = X_sd)),
                      ncol=ncol(X)), dtype="float32")))
    # Severity predictions: only for policies with freq > 0 (simulate always with >0 policies)
    n <- nrow(X_)
    freq_pred <- rep(NA, n)
    # Use correct freq param
    if (fit$freq_dist == "Poisson") {
      lambda_pred <- freq_pred_scaled * freq_sd + freq_mean
      freq_pred <- lambda_pred
    } else if (fit$freq_dist == "ZIP") {
      lambda_pred <- freq_pred_scaled * freq_sd + freq_mean
      freq_pred <- (1 - pi_hat) * lambda_pred
    } else if (fit$freq_dist == "NB") {
      p_pred <- freq_pred_scaled * p_sd + p_mean
      p_pred <- pmin(pmax(p_pred, 1e-8), 1-1e-8)
      freq_pred <- r_hat * (1 - p_pred) / p_pred
    }
    
    # For severity
    X_pol_ <- X_[freq[idx]>0,,drop=FALSE]
    X_pol_scaled <- scale(X_pol_, center = scaler$X_pol_mean, scale = scaler$X_pol_sd)
    sev_pred_scaled <- as.vector(sev_model$predict(
      np$array(matrix(as.numeric(X_pol_scaled), ncol=ncol(X)), dtype="float32")))
    n_pol <- sum(freq[idx]>0)
    sev_pred <- rep(NA, n)
    sev_pred[freq[idx]>0] <- sev_pred_scaled * sev_mean_sd + sev_mean_mean
    
    # Aggregate simulation (slow but clear)
    S_est <- numeric(Msim)
    S_true <- numeric(Msim)
    for (m in seq_len(Msim)) {
      j <- sample(idx, 1)
      if (fit$freq_dist == "Poisson") {
        n_j <- rpois(1, freq_pred[j])
      } else if (fit$freq_dist == "ZIP") {
        n_j <- if(runif(1)<pi_hat) 0 else rpois(1, lambda_pred[j])
      } else if (fit$freq_dist == "NB") {
        mu_j <- freq_pred[j]
        n_j <- rnbinom(1, size=r_hat, mu=mu_j)
      }
      if (n_j == 0) { S_est[m] <- 0 } else {
        if (fit$sev_dist == "Gamma") {
          beta_j <- sev_pred[j]
          S_est[m] <- sum(rgamma(n_j, shape=alpha_hat, scale=beta_j))
        } else {
          mu_log_j <- sev_pred[j]
          S_est[m] <- sum(rlnorm(n_j, meanlog=mu_log_j, sdlog=sigma_hat))
        }
      }
      # True
      j_true <- sample(idx, 1)
      S_true[m] <- sum(sev_list[[j_true]])
    }
    df <- data.frame(
      agg_loss = c(S_est, S_true),
      Type = rep(c("Estimated", "True"), each=Msim)
    )
    p <- ggplot(df, aes(x = agg_loss, color = Type)) +
      geom_density(lwd=1.2, adjust=1.2) +
      labs(title="Aggregate Loss Distribution (Validation Set)", x="Aggregate Loss", y="Density") +
      theme_minimal()
    print(p)
  }
  
  ## -- Main plotting
  plot_freq(train_idx, "Training set")
  plot_sev(train_idx, "Training set")
  plot_freq(val_idx, "Validation set")
  plot_sev(val_idx, "Validation set")
  simulate_aggregate(val_idx)
}
