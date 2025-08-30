fit_zip_gamma <- function(
    X, freq, sev_list,
    kernel = "exponential",      # or "exponential"
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
  require(pscl)
  require(MASS)
  age <- as.numeric(X[,1])
  zip_df <- data.frame(age = age, count = freq)
  age_mean <- mean(zip_df$age)
  age_sd   <- sd(zip_df$age)
  age_scaled <- (zip_df$age - age_mean) / age_sd
  
  # Step 1: Fit ZIP for starting values
  fit_zip <- pscl::zeroinfl(count ~ 1 | 1, data = zip_df, dist = "poisson")
  pi_hat <- plogis(coef(fit_zip)["zero_(Intercept)"])
  lambda_zip <- zip_df$count / (1 - pi_hat)
  lambda_zip[is.na(lambda_zip) | is.infinite(lambda_zip)] <- 0
  
  # Standardized naming: use freq_mean/freq_sd
  freq_mean <- mean(lambda_zip)
  freq_sd   <- sd(lambda_zip)
  lambda_zip_scaled <- (lambda_zip - freq_mean) / freq_sd
  
  # Train/validation split
  n <- nrow(zip_df)
  set.seed(369)
  ix <- sample(n)
  n_val <- as.integer(n * 0.1)
  val_idx   <- as.integer(ix[1:n_val])
  train_idx <- as.integer(ix[(n_val+1):n])
  
  x_train <- np$array(matrix(as.numeric(age_scaled[train_idx]), ncol=1), dtype='float32')
  y_train <- np$array(as.numeric(lambda_zip_scaled[train_idx]), dtype='float32')
  x_val   <- np$array(matrix(as.numeric(age_scaled[val_idx]), ncol=1), dtype='float32')
  y_val   <- np$array(as.numeric(lambda_zip_scaled[val_idx]), dtype='float32')
  
  keras::k_clear_session()
  input_p <- layer_input(shape=1, name="age_p")
  out_p   <- input_p %>%
    layer_dense(units=128, activation="relu") %>%
    layer_dropout(rate=0.15) %>%
    layer_dense(units=32, activation="relu") %>%
    layer_dropout(rate=0.1) %>%
    layer_dense(units=1, activation="linear", name="lambda_zip_scaled")
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
  
  # Per-policy aggregation for severity
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
  
  # Initial psi: predicted lambda for these policies
  lambda_pred_pol_scaled <- as.vector(zip_model$predict(
    np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
  lambda_pred_pol <- lambda_pred_pol_scaled * freq_sd + freq_mean
  lambda_pred_pol_adj <- (1 - pi_hat) * lambda_pred_pol
  psi_pol <- (n_pol - lambda_pred_pol_adj) / sqrt(lambda_pred_pol_adj)
  psi_all2_pol <- np$array(as.numeric(psi_pol), dtype='float32')
  y_gamma_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
  
  # Gamma shape parameter
  sev <- unlist(sev_list)
  alpha_hat <- fitdistr(sev, "gamma")$estimate["shape"]
  
  # --- Kernels ---
  laplace_poisson <- function(lambda, delta) exp(lambda * (exp(-delta) - 1))
  pN0_poisson <- function(lambda) exp(-lambda)
  psi_exp_poisson <- function(N, lambda, delta) {
    LN_delta <- laplace_poisson(lambda, delta)
    p0 <- pN0_poisson(lambda)
    hatL_N <- (LN_delta - p0) / (1 - p0)
    exp(-delta * N) - hatL_N
  }
  laplace_gamma <- function(alpha, beta, gamma) (1 + beta * gamma)^(-alpha)
  phi_exp_gamma <- function(x, alpha, beta, gamma) exp(-gamma * x) - laplace_gamma(alpha, beta, gamma)
  phi_std_gamma <- function(x, mu, sx) (x - mu) / sx
  
  get_phi_vector <- function(x, alpha, beta, kernel, gammaL) {
    if(kernel == "exponential") {
      phi_exp_gamma(x, alpha, beta, gammaL)
    } else {
      phi_std_gamma(x, alpha * beta, sqrt(alpha) * beta)
    }
  }
  get_psi_vector <- function(n, lambda, kernel, delta) {
    if(kernel == "exponential") {
      exp(-delta * n) - exp(-delta * lambda)
    } else {
      (n - lambda) / sqrt(lambda)
    }
  }
  
  phi_exp_gamma_tf <- function(x, alpha, beta, gammaL) {
    tf$math$exp(-gammaL * x) - tf$math$pow(1 + beta * gammaL, -alpha)
  }
  phi_std_gamma_tf <- function(x, mu, sx) (x - mu) / sx
  
  # ---- Main Alternating IFM NN Sarmanov estimation loop ----
  max_iter <- 50
  tol      <- 1e-2
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
    
    lambda_pred_pol_scaled <- as.vector(zip_model$predict(
      np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
    lambda_pred_pol <- lambda_pred_pol_scaled * freq_sd + freq_mean
    lambda_pred_pol_adj <- (1 - pi_hat) * lambda_pred_pol
    beta_pred_pol_scaled <- as.vector(gamma_model$predict(
      np$array(matrix(as.numeric(age_pol_scaled), ncol=1), dtype="float32")))
    beta_pred_pol <- beta_pred_pol_scaled * sev_mean_sd + sev_mean_mean
    
    psi_pol <- get_psi_vector(n_pol, lambda_pred_pol_adj, kernel, delta)
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
    
    psi_all2_pol <- get_psi_vector(n_pol, lambda_pred_pol_adj, kernel, delta)
    y_gamma_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
  }
  
  # --- Final NN predictions (on all ages for aggregate sim) ---
  age_test <- seq(20, 60, by = 1)
  age_test_scaled <- (age_test - age_pol_mean) / age_pol_sd
  x_test_scaled   <- np$array(matrix(as.numeric(age_test_scaled), ncol = 1), dtype = "float32")
  lambda_pred_scaled <- as.vector(zip_model$predict(
    np$array(matrix(as.numeric((age_test - age_mean)/age_sd), ncol=1), dtype="float32")))
  lambda_pred <- lambda_pred_scaled * freq_sd + freq_mean
  lambda_pred_adj <- (1 - pi_hat) * lambda_pred
  
  beta_pred_scaled   <- as.vector(gamma_model$predict(x_test_scaled))
  beta_pred <- beta_pred_scaled * sev_mean_sd + sev_mean_mean
  
  # For aggregate sim, predict on all observed ages (not grid):
  age_all_scaled <- (age - age_pol_mean) / age_pol_sd
  x_all_scaled   <- np$array(matrix(as.numeric(age_all_scaled), ncol = 1), dtype = "float32")
  lambda_pred_nn_scaled <- as.vector(zip_model$predict(
    np$array(matrix(as.numeric((age - age_mean)/age_sd), ncol=1), dtype="float32")))
  lambda_pred_nn <- lambda_pred_nn_scaled * freq_sd + freq_mean
  lambda_pred_nn_adj <- (1 - pi_hat) * lambda_pred_nn
  beta_pred_nn_scaled <- as.vector(gamma_model$predict(x_all_scaled))
  beta_pred_nn <- beta_pred_nn_scaled * sev_mean_sd + sev_mean_mean
  
  return(list(
    zip_model = zip_model,
    gamma_model = gamma_model,
    omega_nn = omega_nn,
    alpha_hat = alpha_hat,
    pi_hat = pi_hat,
    scaler = list(
      age_mean=age_mean, age_sd=age_sd,
      freq_mean=freq_mean, freq_sd=freq_sd,
      age_pol_mean=age_pol_mean, age_pol_sd=age_pol_sd,
      sev_mean_mean=sev_mean_mean, sev_mean_sd=sev_mean_sd
    ),
    lambda_pred = lambda_pred_adj,
    beta_pred = beta_pred,
    lambda_pred_nn = lambda_pred_nn_adj,
    beta_pred_nn = beta_pred_nn,
    kernel = kernel,
    delta = delta,
    gammaL = gammaL
  ))
}
