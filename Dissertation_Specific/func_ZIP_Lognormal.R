fit_zip_lognormal <- function(
    X, freq, sev_list,
    kernel = "standardized",      # or "exponential"
    delta = 0.5,                  # for exponential psi, if needed
    gammaL = 0.5,                 # for exponential phi
    hyper = list(
      freq_units = c(128,32),
      freq_activations = c("relu", "relu"),
      freq_dropout = c(0.15, 0.1),
      sev_units = c(64,32,16,8),
      sev_activations = c("relu","relu","relu","relu"),
      sev_dropout = c(0.15,0.1,0.05,0.05),
      epochs = 250,
      batch_size = 256,
      lr_freq = 6e-3,
      lr_sev = 1e-3
    ),
    verbose = 2
) {
  # --- Frequency (ZIP) network training ---
  age <- as.numeric(X[,1])
  zip_df <- data.frame(age = age, count = freq)
  age_mean <- mean(zip_df$age)
  age_sd   <- sd(zip_df$age)
  age_scaled <- (zip_df$age - age_mean) / age_sd
  
  # Fit a constant ZIP model to get global pi_hat
  library(pscl)
  zip_fit <- zeroinfl(count ~ 1 | 1, data=zip_df, dist="poisson")
  pi_hat <- plogis(coef(zip_fit)["zero_(Intercept)"])
  lambda_hat <- freq / (1 - pi_hat)
  lambda_hat[is.na(lambda_hat) | is.infinite(lambda_hat)] <- 0
  
  # Scale lambda_hat as NN target
  freq_mean <- mean(lambda_hat)
  freq_sd   <- sd(lambda_hat)
  lambda_hat_scaled <- (lambda_hat - freq_mean) / freq_sd
  
  n <- nrow(zip_df)
  set.seed(369)
  ix <- sample(n)
  n_val <- as.integer(n * 0.1)
  val_idx   <- as.integer(ix[1:n_val])
  train_idx <- as.integer(ix[(n_val+1):n])
  
  x_train <- np$array(matrix(as.numeric(age_scaled[train_idx]), ncol=1), dtype='float32')
  y_train <- np$array(as.numeric(lambda_hat_scaled[train_idx]), dtype='float32')
  x_val   <- np$array(matrix(as.numeric(age_scaled[val_idx]), ncol=1), dtype='float32')
  y_val   <- np$array(as.numeric(lambda_hat_scaled[val_idx]), dtype='float32')
  
  keras::k_clear_session()
  input_p <- layer_input(shape=1, name="age_p")
  out_p   <- input_p %>%
    layer_dense(units=128, activation="relu") %>%
    layer_dropout(rate=0.15) %>%
    layer_dense(units=32, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=1, activation="linear", name="lambda_hat_scaled")
  zip_model <- keras_model(inputs=input_p, outputs=out_p)
  zip_model$compile(
    optimizer = optimizer_adam(learning_rate = hyper$lr_freq),
    loss      = "mse"
  )
  history_p <- zip_model$fit(
    x = x_train,
    y = y_train,
    epochs = as.integer(hyper$epochs),
    batch_size = as.integer(hyper$batch_size),
    validation_data = list(x_val, y_val),
    verbose = verbose
  )
  
  # --- Per-policy aggregation for severity ---
  policy_idx <- which(freq > 0)
  age_pol    <- age[policy_idx]
  n_pol      <- freq[policy_idx]
  sev_mean   <- sapply(sev_list[policy_idx], mean) # This is the observed mean severity per policy
  
  age_pol_mean <- mean(age_pol)
  age_pol_sd   <- sd(age_pol)
  age_pol_scaled <- (age_pol - age_pol_mean) / age_pol_sd
  
  # Lognormal sigma (across all claims)
  sev <- unlist(sev_list)
  sigma_hat <- sd(log(sev))
  
  # Implied meanlog per policy (not for likelihood, just for scaling)
  mu_implied <- log(sev_mean) - 0.5 * sigma_hat^2
  mu_implied_mean <- mean(mu_implied)
  mu_implied_sd   <- sd(mu_implied)
  mu_implied_scaled <- (mu_implied - mu_implied_mean) / mu_implied_sd
  
  # Prepare NN data: x = age, y = observed mean severity (not log!)
  xg_pol <- np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype='float32')
  yg_pol <- np$array(as.numeric(sev_mean), dtype='float32') # use mean severity
  
  # Initial psi: use predicted lambda for these policies
  lambda_pred_pol_scaled <- as.vector(zip_model$predict(
    np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
  lambda_pred_pol <- lambda_pred_pol_scaled * freq_sd + freq_mean
  psi_pol <- (n_pol - (1-pi_hat)*lambda_pred_pol) / sqrt((1-pi_hat)*lambda_pred_pol)
  psi_all2_pol <- np$array(as.numeric(psi_pol), dtype='float32')
  y_ln_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
  
  # --- Kernel Functions (unchanged) ---
  laplace_zip <- function(lambda, pi, delta) {
    exp((1-pi)*lambda * (exp(-delta) - 1))
  }
  pN0_zip <- function(lambda, pi) pi + (1-pi) * exp(-lambda)
  psi_exp_zip <- function(N, lambda, pi, delta) {
    LN_delta <- laplace_zip(lambda, pi, delta)
    p0 <- pN0_zip(lambda, pi)
    hatL_N <- (LN_delta - p0) / (1 - p0)
    exp(-delta * N) - hatL_N
  }
  laplace_lognormal <- function(mu, sigma, gammaL) exp(mu * (-gammaL) + 0.5 * sigma^2 * gammaL^2)
  phi_exp_lognormal <- function(x, mu, sigma, gammaL) exp(-gammaL * x) - laplace_lognormal(mu, sigma, gammaL)
  phi_std_lognormal <- function(x, mu, sx) (x - mu) / sx
  
  get_phi_vector <- function(x, mu, sigma, kernel, gammaL) {
    if(kernel == "exponential") {
      phi_exp_lognormal(x, mu, sigma, gammaL)
    } else {
      phi_std_lognormal(x, mu, sigma)
    }
  }
  get_psi_vector <- function(n, lambda, pi, kernel, delta) {
    if(kernel == "exponential") {
      psi_exp_zip(n, lambda, pi, delta)
    } else {
      (n - (1-pi)*lambda) / sqrt((1-pi)*lambda)
    }
  }
  
  phi_exp_lognormal_tf <- function(x, mu, sigma, gammaL) {
    tf$math$exp(-gammaL * x) - tf$math$exp(mu * (-gammaL) + 0.5 * sigma^2 * gammaL^2)
  }
  phi_std_lognormal_tf <- function(x, mu, sx) (x - mu) / sx
  
  # --- Alternating IFM NN-Sarmanov estimation loop ---
  max_iter <- 50
  tol      <- 1e-1
  omega_nn    <- 0.1
  omega_trace <- numeric(max_iter+1)
  omega_trace[1] <- omega_nn
  
  for(loop in 1:max_iter) {
    # Loss function for lognormal with Sarmanov weight
    logn_loss_with_omega <- function(y_true, y_pred) {
      # y_true[, 1] = observed mean severity per policy
      # y_pred[, 1] = predicted meanlog (unscaled)
      sev_mean <- y_true[,1] # observed mean severity, not log!
      psi      <- y_true[,2]
      mu_pred_scaled <- y_pred[,1]
      
      # Undo scaling on mu_pred
      mu_pred <- mu_pred_scaled * mu_implied_sd + mu_implied_mean
      
      # Avoid negatives/zeroes in severity
      sev_mean <- k_maximum(sev_mean, 1e-6)
      
      # Lognormal log-likelihood:
      #   -0.5*log(2pi) - log(sigma) - log(x) - 0.5*((log(x)-mu)^2)/sigma^2
      ll <- -0.5 * k_log(2 * pi) - k_log(sigma_hat) - k_log(sev_mean) -
        0.5 * k_square((k_log(sev_mean) - mu_pred) / sigma_hat)
      
      # Sarmanov term
      if (kernel == "exponential") {
        phi_val <- tf$math$exp(-gammaL * sev_mean) - tf$math$exp(mu_pred * (-gammaL) + 0.5 * sigma_hat^2 * gammaL^2)
      } else {
        phi_val <- (k_log(sev_mean) - mu_pred) / sigma_hat
      }
      sarmanov_weight <- 1 + omega_nn * psi * phi_val
      sarmanov_weight <- k_maximum(sarmanov_weight, 1e-6)
      
      loss <- -(ll + k_log(sarmanov_weight))
      return(loss)
    }
    
    keras::k_clear_session()
    input_g <- layer_input(shape=1, name="age_g")
    out_g <- input_g %>%
      layer_dense(units=64, activation="relu") %>%
      layer_dropout(rate=0.15) %>%
      layer_dense(units=32, activation="relu") %>%
      layer_dropout(rate=0.1) %>%
      layer_dense(units=16, activation="relu") %>%
      layer_dropout(rate=0.05) %>%
      layer_dense(units=8, activation="relu") %>%
      layer_dropout(rate=0.05) %>%
      layer_dense(units=1, activation="linear", name="mu_scaled")
    ln_model <- keras_model(inputs=input_g, outputs=out_g)
    ln_model$compile(
      optimizer = optimizer_adam(learning_rate = hyper$lr_sev),
      loss      = logn_loss_with_omega
    )
    history_g <- ln_model$fit(
      x = xg_pol,
      y = y_ln_pol,
      epochs = as.integer(hyper$epochs),
      batch_size = as.integer(hyper$batch_size),
      verbose = verbose
    )
    
    # ---- Sarmanov parameter re-estimation ----
    lambda_pred_pol_scaled <- as.vector(zip_model$predict(
      np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
    lambda_pred_pol <- lambda_pred_pol_scaled * freq_sd + freq_mean
    mu_pred_pol_scaled <- as.vector(ln_model$predict(
      np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
    mu_pred_pol <- mu_pred_pol_scaled * mu_implied_sd + mu_implied_mean
    
    phi_pol <- get_phi_vector(log(sev_mean), mu_pred_pol, sigma_hat, kernel, gammaL)
    psi_pol <- get_psi_vector(n_pol, lambda_pred_pol, pi_hat, kernel, delta)
    z       <- psi_pol * phi_pol
    
    lower <- if(any(z > 0)) max(-1/z[z > 0]) else -Inf
    upper <- if(any(z < 0)) min(-1/z[z < 0]) else Inf
    lower <- lower + .Machine$double.eps
    upper <- upper - .Machine$double.eps
    
    negll_omega <- function(w){
      W <- 1 + w * psi_pol * phi_pol
      if(any(W <= 0)) return(Inf)
      -sum(log(W))
    }
    opt    <- optimize(negll_omega, interval=c(lower, upper))
    omega_new <- opt$minimum
    omega_trace[loop+1] <- omega_new
    if(verbose > 0) cat("Estimated omega after IFM step:", round(omega_new, 5), "\n")
    if(abs(omega_new - omega_nn) < tol) {
      if(verbose > 0) cat("Converged! Breaking.\n")
      break
    }
    omega_nn <- omega_new
    
    # Re-pack for next iteration
    psi_all2_pol <- get_psi_vector(n_pol, lambda_pred_pol, pi_hat, kernel, delta)
    y_ln_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
  }
  
  # --- Final predictions (grid and all obs) ---
  age_test <- seq(20, 60, by = 1)
  age_test_scaled <- (age_test - age_pol_mean) / age_pol_sd
  x_test_scaled   <- np$array(matrix(as.numeric(age_test_scaled), ncol = 1), dtype = "float32")
  lambda_pred_scaled <- as.vector(zip_model$predict(
    np$array(matrix(as.numeric((age_test - age_mean)/age_sd), ncol=1), dtype="float32")))
  lambda_pred <- lambda_pred_scaled * freq_sd + freq_mean
  mu_pred_scaled   <- as.vector(ln_model$predict(x_test_scaled))
  mu_pred <- mu_pred_scaled * mu_implied_sd + mu_implied_mean
  
  # All observed ages:
  age_all_scaled <- (age - age_pol_mean) / age_pol_sd
  x_all_scaled   <- np$array(matrix(as.numeric(age_all_scaled), ncol = 1), dtype = "float32")
  lambda_pred_nn_scaled <- as.vector(zip_model$predict(
    np$array(matrix(as.numeric((age - age_mean)/age_sd), ncol=1), dtype="float32")))
  lambda_pred_nn <- lambda_pred_nn_scaled * freq_sd + freq_mean
  mu_pred_nn_scaled <- as.vector(ln_model$predict(x_all_scaled))
  mu_pred_nn <- mu_pred_nn_scaled * mu_implied_sd + mu_implied_mean
  
  return(list(
    zip_model = zip_model,
    ln_model = ln_model,
    omega_nn = omega_nn,
    sigma_hat = sigma_hat,
    pi_hat = pi_hat,
    scaler = list(
      age_mean=age_mean, age_sd=age_sd,
      freq_mean=freq_mean, freq_sd=freq_sd,
      age_pol_mean=age_pol_mean, age_pol_sd=age_pol_sd,
      mu_implied_mean=mu_implied_mean, mu_implied_sd=mu_implied_sd
    ),
    lambda_pred = lambda_pred,
    mu_pred = mu_pred,
    lambda_pred_nn = lambda_pred_nn,
    mu_pred_nn = mu_pred_nn,
    kernel = kernel,
    delta = delta,
    gammaL = gammaL
  ))
}
