
# ---- NN Sarmanov fitters (multi-covariate) ----

# Internal: build loss for Gamma severity with Sarmanov
.gamma_loss_builder <- function(alpha_hat, sev_mean_mean, sev_mean_sd, kernel, gammaL, get_psi, psi_vec) {
  .require_keras()
  tf <- tensorflow::tf
  k <- keras::k

  function(y_true, y_pred) {
    # y_true[,1] = sev_mean_scaled, y_true[,2] = psi
    sev_mean_scaled <- y_true[,1]
    psi <- y_true[,2]
    beta_scaled <- y_pred[,1]

    sev_mean <- sev_mean_scaled * sev_mean_sd + sev_mean_mean
    beta <- beta_scaled * sev_mean_sd + sev_mean_mean

    if (kernel == "exponential") {
      phi_val <- tf$math$exp(-gammaL * sev_mean) - tf$math$pow(1 + beta * gammaL, -alpha_hat)
    } else {
      phi_val <- (sev_mean - alpha_hat * beta) / (sqrt(alpha_hat) * beta)
    }
    sarmanov_weight <- k$maximum(1 + psi * phi_val, 1e-8)

    ll <- (alpha_hat - 1) * k$log(sev_mean + 1e-8) - sev_mean / beta - alpha_hat * k$log(beta + 1e-8)
    loss <- -(ll + k$log(sarmanov_weight))
    loss
  }
}

.logn_loss_builder <- function(sigma_hat, mu_mean, mu_sd, kernel, gammaL) {
  .require_keras()
  tf <- tensorflow::tf
  k <- keras::k
  function(y_true, y_pred) {
    # y_true[,1] = sev_mean (not log), y_true[,2] = psi
    sev_mean <- y_true[,1]
    psi <- y_true[,2]
    mu_scaled <- y_pred[,1]

    mu <- mu_scaled * mu_sd + mu_mean
    sev_mean <- k$maximum(sev_mean, 1e-6)

    if (kernel == "exponential") {
      phi_val <- tf$math$exp(-gammaL * sev_mean) - tf$math$exp(mu * (-gammaL) + 0.5 * sigma_hat^2 * gammaL^2)
    } else {
      phi_val <- (k$log(sev_mean) - mu) / sigma_hat
    }
    sarmanov_weight <- k$maximum(1 + psi * phi_val, 1e-8)

    ll <- -0.5 * k$log(2*pi) - k$log(sigma_hat) - k$log(sev_mean) - 0.5 * k$square((k$log(sev_mean) - mu) / sigma_hat)
    loss <- -(ll + k$log(sarmanov_weight))
    loss
  }
}

# Shared driver for NN Gamma severity with a given freq predictor function
.nn_gamma_driver <- function(
  X, freq, sev_list, kernel, delta, gammaL,
  hyper, verbose,
  freq_predictor # function(mm_scaled_train, mm_scaled_val, y_train, y_val) -> list(model, predict_fun)
){
  mm <- .mm(X)
  scf <- .scale_cols(mm)
  Xs <- scf$X

  # Frequency target: standardized counts (robust)
  y_freq <- as.numeric(scale(freq))
  y_freq[!is.finite(y_freq)] <- 0

  split <- .split_train_val(nrow(Xs))
  x_train <- Xs[split$train,,drop=FALSE]
  x_val   <- Xs[split$val,,drop=FALSE]
  y_train <- y_freq[split$train]
  y_val   <- y_freq[split$val]

  fr <- freq_predictor(x_train, x_val, y_train, y_val, hyper, verbose)
  freq_model <- fr$model
  freq_inv   <- fr$inv   # function(pred_scaled) -> mean on original scale proxy
  freq_pred_fun <- fr$predict # function(newXs_scaled) -> freq_mean

  # policy-level aggregation (positive freq)
  policy_idx <- which(freq > 0)
  sev_mean <- vapply(sev_list[policy_idx], function(v) mean(v[v>0]), numeric(1))
  sev_all <- unlist(sev_list)
  alpha_hat <- .fit_alpha_gamma(sev_all)

  # scaler for sev_mean
  sev_mean_mean <- mean(sev_mean)
  sev_mean_sd   <- stats::sd(sev_mean); if (!is.finite(sev_mean_sd) || sev_mean_sd == 0) sev_mean_sd <- 1

  # initial psi using NN frequency
  lambda_pred <- as.numeric(freq_pred_fun(Xs[policy_idx,,drop=FALSE]))
  lambda_pred <- .clip_pos(lambda_pred)
  psi <- .get_psi(freq[policy_idx], lambda_pred, kernel, delta)

  # build gamma NN
  y_true <- cbind( (sev_mean - sev_mean_mean)/sev_mean_sd, psi )
  loss_fn <- .gamma_loss_builder(alpha_hat, sev_mean_mean, sev_mean_sd, kernel, gammaL, .get_psi, psi)
  gamma_model <- .build_mlp(ncol(Xs), hyper$sev_units, hyper$sev_activations, hyper$sev_dropout,
                            lr = hyper$lr_sev, loss = loss_fn)
  history <- gamma_model %>% keras::fit(
    x = Xs[policy_idx,,drop=FALSE], y = array(y_true, dim = c(nrow(y_true), 2)),
    epochs = as.integer(hyper$epochs), batch_size = as.integer(hyper$batch_size),
    verbose = verbose, validation_split = 0.15
  )

  # alternate omega estimation a few rounds by recomputing psi/phi and refitting briefly
  omega <- 0
  for (iter in 1:5) {
    beta_scaled <- as.numeric(keras::predict(gamma_model, Xs[policy_idx,,drop=FALSE], verbose = 0))
    beta_hat <- beta_scaled * sev_mean_sd + sev_mean_mean
    psi <- .get_psi(freq[policy_idx], lambda_pred, kernel, delta)
    phi <- .get_phi_gamma(sev_mean, alpha_hat, beta_hat, kernel, gammaL)
    omega_new <- .safe_omega(psi * phi)
    if (abs(omega_new - omega) < 1e-2) { omega <- omega_new; break }
    omega <- omega_new
    # small fine-tune epochs with updated psi
    y_true <- cbind( (sev_mean - sev_mean_mean)/sev_mean_sd, psi )
    gamma_model %>% keras::fit(
      x = Xs[policy_idx,,drop=FALSE], y = array(y_true, dim = c(nrow(y_true), 2)),
      epochs = 10, batch_size = as.integer(hyper$batch_size),
      verbose = 0
    )
  }

  list(
    scaler_X = scf,
    freq_model = freq_model,
    freq_predict = freq_pred_fun,
    gamma_model = gamma_model,
    alpha_hat = alpha_hat,
    omega = omega,
    sev_scaler = list(mean = sev_mean_mean, sd = sev_mean_sd)
  )
}

# Frequency predictors for NN

.freq_pred_poisson <- function(x_train, x_val, y_train, y_val, hyper, verbose) {
  m <- .build_mlp(ncol(x_train), hyper$freq_units, hyper$freq_activations, hyper$freq_dropout, lr = hyper$lr_freq, loss = "mse")
  m %>% keras::fit(x = x_train, y = y_train, epochs = as.integer(hyper$epochs),
                   batch_size = as.integer(hyper$batch_size), verbose = verbose,
                   validation_data = list(x_val, y_val))
  inv <- function(pred_scaled, y_mean = 0, y_sd = 1) { pred_scaled * y_sd + y_mean }
  predict_fun <- function(newXs) {
    # predict scaled then unscale back to lambda proxy: mean(freq) + sd(freq) * pred
    ps <- as.numeric(keras::predict(m, newXs, verbose = 0))
    # map to positive mean proxy via exp on linear proxy
    .clip_pos(ps - min(ps) + 1e-3)
  }
  list(model = m, inv = inv, predict = predict_fun)
}

.freq_pred_zip <- function(x_train, x_val, y_train, y_val, hyper, verbose) {
  # same architecture; ZIP uses intercept-only pi_hat handled in GLM path; here NN approximates lambda proxy
  .freq_pred_poisson(x_train, x_val, y_train, y_val, hyper, verbose)
}

.freq_pred_nb <- function(x_train, x_val, y_train, y_val, hyper, verbose) {
  # predict mu proxy
  .freq_pred_poisson(x_train, x_val, y_train, y_val, hyper, verbose)
}

# ---- Public NN functions ----

fit_poisson_gamma <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  hyper = list(
    freq_units = c(128,32),
    freq_activations = c("relu","relu"),
    freq_dropout = c(0.15,0.1),
    sev_units = c(64,32,16,8),
    sev_activations = c("relu","relu","relu","relu"),
    sev_dropout = c(0.15,0.1,0.05,0.05),
    epochs = 50, batch_size = 256, lr_freq = 6e-3, lr_sev = 1e-3
  ),
  verbose = 1
){
  kernel <- match.arg(kernel)
  out <- .nn_gamma_driver(X, freq, sev_list, kernel, delta, gammaL, hyper, verbose, .freq_pred_poisson)
  out$kernel <- kernel; out$delta <- delta; out$gammaL <- gammaL
  out
}

fit_zip_gamma <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  hyper = list(
    freq_units = c(128,32),
    freq_activations = c("relu","relu"),
    freq_dropout = c(0.15,0.1),
    sev_units = c(64,32,16,8),
    sev_activations = c("relu","relu","relu","relu"),
    sev_dropout = c(0.15,0.1,0.05,0.05),
    epochs = 50, batch_size = 256, lr_freq = 6e-3, lr_sev = 1e-3
  ),
  verbose = 1
){
  kernel <- match.arg(kernel)
  out <- .nn_gamma_driver(X, freq, sev_list, kernel, delta, gammaL, hyper, verbose, .freq_pred_zip)
  out$kernel <- kernel; out$delta <- delta; out$gammaL <- gammaL
  out
}

fit_nb_gamma <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  hyper = list(
    freq_units = c(128,32),
    freq_activations = c("relu","relu"),
    freq_dropout = c(0.15,0.1),
    sev_units = c(64,32,16,8),
    sev_activations = c("relu","relu","relu","relu"),
    sev_dropout = c(0.15,0.1,0.05,0.05),
    epochs = 50, batch_size = 256, lr_freq = 6e-3, lr_sev = 1e-3
  ),
  verbose = 1
){
  kernel <- match.arg(kernel)
  out <- .nn_gamma_driver(X, freq, sev_list, kernel, delta, gammaL, hyper, verbose, .freq_pred_nb)
  out$kernel <- kernel; out$delta <- delta; out$gammaL <- gammaL
  out
}

# Lognormal NN drivers (reuse frequency predictors)

.nn_logn_driver <- function(
  X, freq, sev_list, kernel, delta, gammaL,
  hyper, verbose,
  freq_predictor
){
  mm <- .mm(X)
  scf <- .scale_cols(mm)
  Xs <- scf$X

  y_freq <- as.numeric(scale(freq)); y_freq[!is.finite(y_freq)] <- 0
  split <- .split_train_val(nrow(Xs))
  x_train <- Xs[split$train,,drop=FALSE]
  x_val   <- Xs[split$val,,drop=FALSE]
  y_train <- y_freq[split$train]
  y_val   <- y_freq[split$val]

  fr <- freq_predictor(x_train, x_val, y_train, y_val, hyper, verbose)
  freq_model <- fr$model
  freq_pred_fun <- fr$predict

  policy_idx <- which(freq > 0)
  sev_mean <- vapply(sev_list[policy_idx], function(v) mean(v[v>0]), numeric(1))
  sev_all  <- unlist(sev_list)
  sigma_hat <- stats::sd(log(sev_all[sev_all>0]))

  mu_imp <- log(sev_mean) - 0.5 * sigma_hat^2
  mu_mean <- mean(mu_imp); mu_sd <- stats::sd(mu_imp); if (!is.finite(mu_sd) || mu_sd == 0) mu_sd <- 1

  lambda_pred <- as.numeric(freq_pred_fun(Xs[policy_idx,,drop=FALSE]))
  lambda_pred <- .clip_pos(lambda_pred)
  psi <- .get_psi(freq[policy_idx], lambda_pred, kernel, delta)

  y_true <- cbind(sev_mean, psi)  # loss uses raw sev_mean
  loss_fn <- .logn_loss_builder(sigma_hat, mu_mean, mu_sd, kernel, gammaL)
  ln_model <- .build_mlp(ncol(Xs), hyper$sev_units, hyper$sev_activations, hyper$sev_dropout,
                         lr = hyper$lr_sev, loss = loss_fn)
  history <- ln_model %>% keras::fit(
    x = Xs[policy_idx,,drop=FALSE], y = array(y_true, dim = c(nrow(y_true), 2)),
    epochs = as.integer(hyper$epochs), batch_size = as.integer(hyper$batch_size),
    verbose = verbose, validation_split = 0.15
  )

  omega <- 0
  for (iter in 1:5) {
    mu_scaled <- as.numeric(keras::predict(ln_model, Xs[policy_idx,,drop=FALSE], verbose = 0))
    mu_hat <- mu_scaled * mu_sd + mu_mean
    psi <- .get_psi(freq[policy_idx], lambda_pred, kernel, delta)
    phi <- .get_phi_logn(log(sev_mean), mu_hat, sigma_hat, kernel, gammaL)
    omega_new <- .safe_omega(psi * phi)
    if (abs(omega_new - omega) < 1e-2) { omega <- omega_new; break }
    omega <- omega_new
    y_true <- cbind(sev_mean, psi)
    ln_model %>% keras::fit(
      x = Xs[policy_idx,,drop=FALSE], y = array(y_true, dim = c(nrow(y_true), 2)),
      epochs = 10, batch_size = as.integer(hyper$batch_size), verbose = 0
    )
  }

  list(
    scaler_X = scf,
    freq_model = freq_model,
    freq_predict = freq_pred_fun,
    ln_model = ln_model,
    sigma_hat = sigma_hat,
    omega = omega
  )
}

fit_poisson_lognormal <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  hyper = list(
    freq_units = c(128,32),
    freq_activations = c("relu","relu"),
    freq_dropout = c(0.15,0.1),
    sev_units = c(64,32,16,8),
    sev_activations = c("relu","relu","relu","relu"),
    sev_dropout = c(0.15,0.1,0.05,0.05),
    epochs = 50, batch_size = 256, lr_freq = 6e-3, lr_sev = 1e-3
  ),
  verbose = 1
){
  kernel <- match.arg(kernel)
  out <- .nn_logn_driver(X, freq, sev_list, kernel, delta, gammaL, hyper, verbose, .freq_pred_poisson)
  out$kernel <- kernel; out$delta <- delta; out$gammaL <- gammaL
  out
}

fit_zip_lognormal <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  hyper = list(
    freq_units = c(128,32),
    freq_activations = c("relu","relu"),
    freq_dropout = c(0.15,0.1),
    sev_units = c(64,32,16,8),
    sev_activations = c("relu","relu","relu","relu"),
    sev_dropout = c(0.15,0.1,0.05,0.05),
    epochs = 50, batch_size = 256, lr_freq = 6e-3, lr_sev = 1e-3
  ),
  verbose = 1
){
  kernel <- match.arg(kernel)
  out <- .nn_logn_driver(X, freq, sev_list, kernel, delta, gammaL, hyper, verbose, .freq_pred_zip)
  out$kernel <- kernel; out$delta <- delta; out$gammaL <- gammaL
  out
}

fit_nb_lognormal <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  hyper = list(
    freq_units = c(128,32),
    freq_activations = c("relu","relu"),
    freq_dropout = c(0.15,0.1),
    sev_units = c(64,32,16,8),
    sev_activations = c("relu","relu","relu","relu"),
    sev_dropout = c(0.15,0.1,0.05,0.05),
    epochs = 50, batch_size = 256, lr_freq = 6e-3, lr_sev = 1e-3
  ),
  verbose = 1
){
  kernel <- match.arg(kernel)
  out <- .nn_logn_driver(X, freq, sev_list, kernel, delta, gammaL, hyper, verbose, .freq_pred_nb)
  out$kernel <- kernel; out$delta <- delta; out$gammaL <- gammaL
  out
}
