fit_nb_gamma <- function(
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
  require(MASS)
  age <- as.numeric(X[,1])
  nb_df <- data.frame(age = age, count = freq)
  age_mean <- mean(nb_df$age)
  age_sd   <- sd(nb_df$age)
  age_scaled <- (nb_df$age - age_mean) / age_sd
  
  # --- Estimate NB parameters globally ---
  fit_nb <- MASS::fitdistr(freq, "Negative Binomial")
  r_hat <- fit_nb$estimate["size"]
  mu_hat <- fit_nb$estimate["mu"]
  
  # For NB: E[N] = r*(1-p)/p => p = r/(r+E[N])
  p_i <- r_hat / (r_hat + freq)
  p_i[freq == 0] <- 1 - 1e-8 # numerical safety
  
  # Scaling for p_i
  p_mean <- mean(p_i)
  p_sd   <- sd(p_i)
  p_scaled <- (p_i - p_mean) / p_sd
  
  # --- Train/validation split ---
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
  
  sev_mean_mean <- mean(sev_mean)
  sev_mean_sd   <- sd(sev_mean)
  sev_mean_scaled <- (sev_mean - sev_mean_mean) / sev_mean_sd
  
  xg_pol <- np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype='float32')
  yg_pol <- np$array(as.numeric(sev_mean_scaled), dtype='float32')
  
  # Initial psi using NN prediction for p
  p_pred_pol_scaled <- as.vector(nb_model$predict(
    np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
  p_pred_pol <- p_pred_pol_scaled * p_sd + p_mean
  p_pred_pol <- pmin(pmax(p_pred_pol, 1e-8), 1-1e-8) # numerical safety
  freq_pred_pol <- r_hat * (1 - p_pred_pol) / p_pred_pol
  psi_pol <- (n_pol - freq_pred_pol) / sqrt(freq_pred_pol)
  psi_all2_pol <- np$array(as.numeric(psi_pol), dtype='float32')
  y_gamma_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
  
  # Gamma shape parameter
  sev <- unlist(sev_list)
  alpha_hat <- fitdistr(sev, "gamma")$estimate["shape"]
  
  # --- Kernel functions ---
  phi_exp_gamma_tf <- function(x, alpha, beta, gammaL) {
    tf$math$exp(-gammaL * x) - tf$math$pow(1 + beta * gammaL, -alpha)
  }
  phi_std_gamma_tf <- function(x, mu, sx) (x - mu) / sx
  
  get_phi_vector <- function(x, alpha, beta, kernel, gammaL) {
    if(kernel == "exponential") {
      exp(-gammaL * x) - (1 + beta * gammaL)^(-alpha)
    } else {
      (x - alpha * beta) / (sqrt(alpha) * beta)
    }
  }
  get_psi_vector <- function(n, freq_pred, kernel, delta) {
    if(kernel == "exponential") {
      exp(-delta * n) - exp(-delta * freq_pred)
    } else {
      (n - freq_pred) / sqrt(freq_pred)
    }
  }
  
  # ---- Main Alternating IFM NN Sarmanov estimation loop ----
  max_iter <- 50
  tol      <- 1e-1
  omega_nn    <- 0.1
  omega_trace <- numeric(max_iter+1)
  omega_trace[1] <- omega_nn
  
  for(loop in 1:max_iter) {
    gamma_loss_with_omega <- function(y_true, y_pred) {
      sev_mean_scaled  <- y_true[,1]
      psi              <- y_true[,2]
      beta_scaled      <- y_pred[,1]
      sev_mean <- sev_mean_scaled * sev_mean_sd + sev_mean_mean
      beta     <- beta_scaled * sev_mean_sd + sev_mean_mean
      if (kernel == "exponential") {
        phi_val <- phi_exp_gamma_tf(sev_mean, alpha_hat, beta, gammaL)
      } else {
        phi_val <- phi_std_gamma_tf(sev_mean, alpha_hat * beta, sqrt(alpha_hat) * beta)
      }
      sarmanov_weight <- 1 + omega_nn * psi * phi_val
      sarmanov_weight <- k_maximum(sarmanov_weight, 1e-8)
      ll <- (alpha_hat-1)*k_log(sev_mean + 1e-8) - sev_mean/beta -
        alpha_hat*k_log(beta)
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
      layer_dense(units=1, activation="linear", name="beta_scaled")
    gamma_model <- keras_model(inputs=input_g, outputs=out_g)
    gamma_model$compile(
      optimizer = optimizer_adam(learning_rate = hyper$lr_sev),
      loss      = gamma_loss_with_omega
    )
    history_g <- gamma_model$fit(
      x = xg_pol,
      y = y_gamma_pol,
      epochs = as.integer(hyper$epochs),
      batch_size = as.integer(hyper$batch_size),
      verbose = verbose
    )
    
    p_pred_pol_scaled <- as.vector(nb_model$predict(
      np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
    p_pred_pol <- p_pred_pol_scaled * p_sd + p_mean
    p_pred_pol <- pmin(pmax(p_pred_pol, 1e-8), 1-1e-8)
    freq_pred_pol <- r_hat * (1 - p_pred_pol) / p_pred_pol
    beta_pred_pol_scaled <- as.vector(gamma_model$predict(
      np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
    beta_pred_pol <- beta_pred_pol_scaled * sev_mean_sd + sev_mean_mean
    
    psi_pol <- get_psi_vector(n_pol, freq_pred_pol, kernel, delta)
    phi_pol <- get_phi_vector(sev_mean, alpha_hat, beta_pred_pol, kernel, gammaL)
    
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
    y_gamma_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
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
  beta_pred_scaled   <- as.vector(gamma_model$predict(x_test_scaled))
  beta_pred <- beta_pred_scaled * sev_mean_sd + sev_mean_mean
  
  # For aggregate sim, predict on all observed ages (not grid):
  age_all_scaled <- (age - age_pol_mean) / age_pol_sd
  x_all_scaled   <- np$array(matrix(as.numeric(age_all_scaled), ncol = 1), dtype = "float32")
  p_pred_nn_scaled <- as.vector(nb_model$predict(
    np$array(matrix(as.numeric((age - age_mean)/age_sd), ncol=1), dtype="float32")))
  p_pred_nn <- p_pred_nn_scaled * p_sd + p_mean
  p_pred_nn <- pmin(pmax(p_pred_nn, 1e-8), 1-1e-8)
  freq_pred_nn <- r_hat * (1 - p_pred_nn) / p_pred_nn
  beta_pred_nn_scaled <- as.vector(gamma_model$predict(x_all_scaled))
  beta_pred_nn <- beta_pred_nn_scaled * sev_mean_sd + sev_mean_mean
  
  return(list(
    nb_model = nb_model,
    gamma_model = gamma_model,
    omega_nn = omega_nn,
    alpha_hat = alpha_hat,
    r_hat = r_hat,
    scaler = list(
      age_mean=age_mean, age_sd=age_sd,
      mu_mean=mu_hat, mu_sd=sd(freq),
      p_mean=p_mean, p_sd=p_sd,
      age_pol_mean=age_pol_mean, age_pol_sd=age_pol_sd,
      sev_mean_mean=sev_mean_mean, sev_mean_sd=sev_mean_sd
    ),
    freq_pred = freq_pred,
    beta_pred = beta_pred,
    freq_pred_nn = freq_pred_nn,
    beta_pred_nn = beta_pred_nn,
    kernel = kernel,
    delta = delta,
    gammaL = gammaL
  ))
}
