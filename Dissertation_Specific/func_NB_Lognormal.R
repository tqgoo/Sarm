fit_nb_lognormal <- function(
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
  # --- Frequency (NB) network training ---
  age <- as.numeric(X[,1])
  nb_df <- data.frame(age = age, count = freq)
  age_mean <- mean(nb_df$age)
  age_sd   <- sd(nb_df$age)
  age_scaled <- (nb_df$age - age_mean) / age_sd
  
  # NB parameter estimation for global r_hat, mu_hat
  library(MASS)
  nb_fit <- fitdistr(freq, "Negative Binomial")
  r_hat <- nb_fit$estimate["size"]
  mu_hat <- nb_fit$estimate["mu"]
  
  # Compute implied mean p_i per policy, use as regression target
  p_i <- r_hat / (r_hat + freq)
  p_i[freq == 0] <- 1 - 1e-8 # numerical safety
  p_mean <- mean(p_i)
  p_sd   <- sd(p_i)
  p_scaled <- (p_i - p_mean) / p_sd
  
  n <- nrow(nb_df)
  set.seed(369)
  ix <- sample(n)
  n_val <- as.integer(n * 0.1)
  val_idx   <- as.integer(ix[1:n_val])
  train_idx <- as.integer(ix[(n_val+1):n])
  
  x_train <- np$array(matrix(as.numeric(age_scaled[train_idx]), ncol=1), dtype='float32')
  y_train <- np$array(as.numeric(p_scaled[train_idx]), dtype='float32')
  x_val   <- np$array(matrix(as.numeric(age_scaled[val_idx]), ncol=1), dtype='float32')
  y_val   <- np$array(as.numeric(p_scaled[val_idx]), dtype='float32')
  
  keras::k_clear_session()
  input_p <- layer_input(shape=1, name="age_p")
  out_p   <- input_p %>%
    layer_dense(units=128, activation="relu") %>%
    layer_dropout(rate=0.15) %>%
    layer_dense(units=32, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=1, activation="linear", name="p_scaled")
  nb_model <- keras_model(inputs=input_p, outputs=out_p)
  nb_model$compile(
    optimizer = optimizer_adam(learning_rate = hyper$lr_freq),
    loss      = "mse"
  )
  history_p <- nb_model$fit(
    x = x_train,
    y = y_train,
    epochs = as.integer(hyper$epochs),
    batch_size = as.integer(hyper$batch_size),
    validation_data = list(x_val, y_val),
    verbose = verbose
  )
  
  # ---- Per-policy aggregation for severity ----
  policy_idx <- which(freq > 0)
  age_pol    <- age[policy_idx]
  n_pol      <- freq[policy_idx]
  sev_mean   <- sapply(sev_list[policy_idx], mean)
  
  age_pol_mean <- mean(age_pol)
  age_pol_sd   <- sd(age_pol)
  age_pol_scaled <- (age_pol - age_pol_mean) / age_pol_sd
  
  # Lognormal sigma (across all claims)
  sev <- unlist(sev_list)
  sigma_hat <- sd(log(sev))
  
  # Implied meanlog per policy (for scaling only)
  mu_implied <- log(sev_mean) - 0.5 * sigma_hat^2
  mu_implied_mean <- mean(mu_implied)
  mu_implied_sd   <- sd(mu_implied)
  mu_implied_scaled <- (mu_implied - mu_implied_mean) / mu_implied_sd
  
  # NN for severity: x = age, y = observed mean severity (not meanlog)
  xg_pol <- np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype='float32')
  yg_pol <- np$array(as.numeric(sev_mean), dtype='float32') # Use mean severity
  
  # Initial psi: use predicted p for these policies
  p_pred_pol_scaled <- as.vector(nb_model$predict(
    np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
  p_pred_pol <- p_pred_pol_scaled * p_sd + p_mean
  p_pred_pol <- pmin(pmax(p_pred_pol, 1e-8), 1-1e-8)
  freq_pred_pol <- r_hat * (1 - p_pred_pol) / p_pred_pol
  psi_pol <- (n_pol - freq_pred_pol) / sqrt(freq_pred_pol)
  psi_all2_pol <- np$array(as.numeric(psi_pol), dtype='float32')
  y_ln_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
  
  # --- Kernels (standardized or exponential) ---
  phi_exp_lognormal <- function(x, mu, sigma, gammaL)
    exp(-gammaL * x) - exp(mu * (-gammaL) + 0.5 * sigma^2 * gammaL^2)
  phi_std_lognormal <- function(x, mu, sx) (x - mu) / sx
  
  get_phi_vector <- function(x, mu, sigma, kernel, gammaL) {
    if(kernel == "exponential") {
      phi_exp_lognormal(x, mu, sigma, gammaL)
    } else {
      phi_std_lognormal(x, mu, sigma)
    }
  }
  get_psi_vector <- function(n, freq_pred, kernel, delta) {
    if(kernel == "exponential") {
      exp(-delta * n) - exp(-delta * freq_pred)
    } else {
      (n - freq_pred) / sqrt(freq_pred)
    }
  }
  
  phi_exp_lognormal_tf <- function(x, mu, sigma, gammaL) {
    tf$math$exp(-gammaL * x) - tf$math$exp(mu * (-gammaL) + 0.5 * sigma^2 * gammaL^2)
  }
  phi_std_lognormal_tf <- function(x, mu, sx) (x - mu) / sx
  
  # ---- Main Alternating IFM NN Sarmanov estimation loop ----
  max_iter <- 50
  tol      <- 1e-1
  omega_nn    <- 0.1
  omega_trace <- numeric(max_iter+1)
  omega_trace[1] <- omega_nn
  
  for(loop in 1:max_iter) {
    logn_loss_with_omega <- function(y_true, y_pred) {
      sev_mean <- y_true[,1]    # OBSERVED mean severity per policy
      psi      <- y_true[,2]
      mu_pred_scaled <- y_pred[,1]
      
      mu_pred <- mu_pred_scaled * mu_implied_sd + mu_implied_mean
      sev_mean <- k_maximum(sev_mean, 1e-6)
      
      # Lognormal log-likelihood
      ll <- -0.5 * k_log(2 * pi) - k_log(sigma_hat) - k_log(sev_mean) -
        0.5 * k_square((k_log(sev_mean) - mu_pred) / sigma_hat)
      
      # Sarmanov kernel
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
    
    p_pred_pol_scaled <- as.vector(nb_model$predict(
      np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
    p_pred_pol <- p_pred_pol_scaled * p_sd + p_mean
    p_pred_pol <- pmin(pmax(p_pred_pol, 1e-8), 1-1e-8)
    freq_pred_pol <- r_hat * (1 - p_pred_pol) / p_pred_pol
    mu_pred_pol_scaled <- as.vector(ln_model$predict(
      np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
    mu_pred_pol <- mu_pred_pol_scaled * mu_implied_sd + mu_implied_mean
    
    phi_pol <- get_phi_vector(log(sev_mean), mu_pred_pol, sigma_hat, kernel, gammaL)
    psi_pol <- get_psi_vector(n_pol, freq_pred_pol, kernel, delta)
    
    negll_omega <- function(w){
      W <- 1 + w * psi_pol * phi_pol
      if(any(W <= 0)) return(Inf)
      -sum(log(W))
    }
    z     <- psi_pol * phi_pol
    lower <- if(any(z > 0)) max(-1/z[z > 0]) else -Inf
    upper <- if(any(z < 0)) min(-1/z[z < 0]) else Inf
    lower <- lower + .Machine$double.eps
    upper <- upper - .Machine$double.eps
    opt    <- optimize(negll_omega, interval=c(lower, upper))
    omega_new <- opt$minimum
    omega_trace[loop+1] <- omega_new
    if(verbose > 0) cat("Estimated omega after IFM step:", round(omega_new, 5), "\n")
    if(abs(omega_new - omega_nn) < tol) {
      if(verbose > 0) cat("Converged! Breaking.\n")
      break
    }
    omega_nn <- omega_new
    
    psi_all2_pol <- get_psi_vector(n_pol, freq_pred_pol, kernel, delta)
    y_ln_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
  }
  
  # --- Final NN predictions (on all ages for aggregate sim) ---
  age_test <- seq(20, 60, by = 1)
  age_test_scaled <- (age_test - age_pol_mean) / age_pol_sd
  x_test_scaled   <- np$array(matrix(as.numeric(age_test_scaled), ncol = 1), dtype = "float32")
  p_pred_scaled <- as.vector(nb_model$predict(
    np$array(matrix(as.numeric((age_test - age_mean)/age_sd), ncol=1), dtype="float32")))
  p_pred <- p_pred_scaled * p_sd + p_mean
  p_pred <- pmin(pmax(p_pred, 1e-8), 1-1e-8)
  freq_pred <- r_hat * (1 - p_pred) / p_pred
  mu_pred_scaled   <- as.vector(ln_model$predict(x_test_scaled))
  mu_pred <- mu_pred_scaled * mu_implied_sd + mu_implied_mean
  
  # For aggregate sim, predict on all observed ages (not grid):
  age_all_scaled <- (age - age_pol_mean) / age_pol_sd
  x_all_scaled   <- np$array(matrix(as.numeric(age_all_scaled), ncol = 1), dtype = "float32")
  p_pred_nn_scaled <- as.vector(nb_model$predict(
    np$array(matrix(as.numeric((age - age_mean)/age_sd), ncol=1), dtype="float32")))
  p_pred_nn <- p_pred_nn_scaled * p_sd + p_mean
  p_pred_nn <- pmin(pmax(p_pred_nn, 1e-8), 1-1e-8)
  freq_pred_nn <- r_hat * (1 - p_pred_nn) / p_pred_nn
  mu_pred_nn_scaled <- as.vector(ln_model$predict(x_all_scaled))
  mu_pred_nn <- mu_pred_nn_scaled * mu_implied_sd + mu_implied_mean
  
  return(list(
    nb_model = nb_model,
    ln_model = ln_model,
    omega_nn = omega_nn,
    sigma_hat = sigma_hat,
    r_hat = r_hat,
    scaler = list(
      age_mean=age_mean, age_sd=age_sd,
      p_mean=p_mean, p_sd=p_sd,
      age_pol_mean=age_pol_mean, age_pol_sd=age_pol_sd,
      mu_implied_mean=mu_implied_mean, mu_implied_sd=mu_implied_sd
    ),
    freq_pred = freq_pred,
    mu_pred = mu_pred,
    freq_pred_nn = freq_pred_nn,
    mu_pred_nn = mu_pred_nn,
    kernel = kernel,
    delta = delta,
    gammaL = gammaL
  ))
}
