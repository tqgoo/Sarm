fit_sarmanov_nn <- function(
    freq_dist = c("Poisson", "ZIP", "NB"),
    sev_dist  = c("Gamma", "Lognormal"),
    X, freq, sev_list,
    kernel_type = c("standardized", "exponential"),
    kernel_gamma = 1,
    hyper = list(
      freq_units = c(128,32),
      freq_activations = c("relu", "relu"),
      freq_dropout = c(0.1, 0.1),
      sev_units = c(64,32,16,8),
      sev_activations = c("relu","relu","relu","relu"),
      sev_dropout = c(0.1,0.1,0.05,0.05),
      epochs = 150,
      batch_size = 256,
      lr_freq = 6e-3,
      lr_sev = 1e-3
    ),
    verbose=2
) {
  freq_dist <- match.arg(freq_dist)
  sev_dist  <- match.arg(sev_dist)
  kernel_type <- match.arg(kernel_type)
  n <- nrow(X)
  p <- ncol(X)
  X <- as.matrix(X)
  
  stopifnot(length(hyper$freq_units) == length(hyper$freq_activations),
            length(hyper$freq_units) == length(hyper$freq_dropout),
            length(hyper$sev_units)  == length(hyper$sev_activations),
            length(hyper$sev_units)  == length(hyper$sev_dropout))
  
  # --- Standardize predictors
  X_mean <- colMeans(X)
  X_sd   <- apply(X, 2, sd)
  X_scaled <- scale(X, center=X_mean, scale=X_sd)
  
  # --- Prepare frequency NN target
  if (freq_dist == "Poisson") {
    y_freq <- freq
    freq_mean <- mean(y_freq)
    freq_sd   <- sd(y_freq)
    freq_scaled <- (y_freq - freq_mean) / freq_sd
  }
  if (freq_dist == "ZIP") {
    library(pscl)
    fit_zip <- zeroinfl(freq ~ 1 | 1, dist = "poisson")
    pi_hat <- plogis(coef(fit_zip)["zero_(Intercept)"])
    lambda_hat <- freq / (1 - pi_hat)
    lambda_hat[is.na(lambda_hat) | is.infinite(lambda_hat)] <- 0
    y_freq <- lambda_hat
    freq_mean <- mean(y_freq)
    freq_sd   <- sd(y_freq)
    freq_scaled <- (y_freq - freq_mean) / freq_sd
  }
  if (freq_dist == "NB") {
    fit_nb <- MASS::fitdistr(freq, "Negative Binomial")
    r_hat <- fit_nb$estimate["size"]
    p_i <- r_hat / (r_hat + freq)
    p_i[freq == 0] <- 1 - 1e-8
    p_mean <- mean(p_i)
    p_sd   <- sd(p_i)
    p_scaled <- (p_i - p_mean) / p_sd
    y_freq <- p_scaled
  }
  
  # --- Train/validation split
  n <- nrow(X)
  ix <- sample(n)
  n_val <- as.integer(n * 0.1)
  val_idx   <- as.integer(ix[1:n_val])
  train_idx <- as.integer(ix[(n_val+1):n])
  x_train <- np$array(matrix(as.numeric(X_scaled[train_idx,]), ncol=p), dtype='float32')
  x_val   <- np$array(matrix(as.numeric(X_scaled[val_idx,]), ncol=p), dtype='float32')
  if (freq_dist == "NB") {
    y_train <- np$array(as.numeric(y_freq[train_idx]), dtype='float32')
    y_val   <- np$array(as.numeric(y_freq[val_idx]), dtype='float32')
  } else {
    y_train <- np$array(as.numeric(freq_scaled[train_idx]), dtype='float32')
    y_val   <- np$array(as.numeric(freq_scaled[val_idx]), dtype='float32')
  }
  
  # --- Build Frequency NN
  keras::k_clear_session()
  input_p <- layer_input(shape=p, name="freq_cov")
  out_p <- input_p
  for (i in seq_along(hyper$freq_units)) {
    out_p <- out_p %>%
      layer_dense(units=hyper$freq_units[i], activation=hyper$freq_activations[i])
    if (hyper$freq_dropout[i] > 0)
      out_p <- out_p %>% layer_dropout(rate=hyper$freq_dropout[i])
  }
  out_p <- out_p %>% layer_dense(units=1, activation="linear", name="freq_scaled")
  freq_model <- keras_model(inputs=input_p, outputs=out_p)
  freq_model$compile(optimizer = optimizer_adam(learning_rate = hyper$lr_freq), loss = "mse")
  freq_model$fit(
    x = x_train, y = y_train,
    epochs = as.integer(hyper$epochs), batch_size = as.integer(hyper$batch_size),
    validation_data = list(x_val, y_val), verbose = verbose
  )
  
  # -- Per-policy data for severity: only those with freq > 0
  policy_idx <- which(freq > 0)
  X_pol    <- X[policy_idx,,drop=FALSE]
  n_pol    <- freq[policy_idx]
  sev_mean <- sapply(sev_list[policy_idx], mean)
  
  X_pol_mean <- colMeans(X_pol)
  X_pol_sd   <- apply(X_pol, 2, sd)
  X_pol_scaled <- scale(X_pol, center=X_pol_mean, scale=X_pol_sd)
  sev_mean_mean <- mean(sev_mean)
  sev_mean_sd   <- sd(sev_mean)
  sev_mean_scaled <- (sev_mean - sev_mean_mean) / sev_mean_sd
  
  xg_pol <- np$array(matrix(as.numeric(X_pol_scaled), ncol=p), dtype='float32')
  yg_pol <- np$array(as.numeric(sev_mean_scaled), dtype='float32')
  
  # Estimate assumed identical parameters
  if (sev_dist == "Gamma") alpha_hat <- fitdistr(unlist(sev_list), "gamma")$estimate["shape"]
  if (sev_dist == "Lognormal") sigma_hat <- sd(log(unlist(sev_list)))
  
  # --- Frequency NN prediction for policies
  freq_pred_pol_scaled <- as.vector(freq_model$predict(
    np$array(matrix(as.numeric(X_pol_scaled), ncol=p), dtype="float32")))
  
  if (freq_dist == "Poisson") {
    lambda_pred_pol <- freq_pred_pol_scaled * freq_sd + freq_mean
    freq_pred_pol <- lambda_pred_pol
    E_N <- freq_pred_pol
    Var_N <- freq_pred_pol
  } else if (freq_dist == "ZIP") {
    lambda_pred_pol <- freq_pred_pol_scaled * freq_sd + freq_mean
    freq_pred_pol <- (1 - pi_hat) * lambda_pred_pol
    E_N <- freq_pred_pol
    Var_N <- (1 - pi_hat) * lambda_pred_pol * (1 + pi_hat * lambda_pred_pol)
  } else if (freq_dist == "NB") {
    p_pred_pol <- freq_pred_pol_scaled * p_sd + p_mean
    p_pred_pol <- pmin(pmax(p_pred_pol, 1e-8), 1-1e-8)
    freq_pred_pol <- r_hat * (1 - p_pred_pol) / p_pred_pol
    E_N <- freq_pred_pol
    Var_N <- freq_pred_pol + (freq_pred_pol^2) / r_hat
  }
  
  # ---- Sarmanov Kernels ----
  if (kernel_type == "exponential") {
    ln_delta_num <- mean(exp(-kernel_gamma * n_pol))
    p0 <- mean(n_pol == 0)
    ln_delta_den <- 1 - p0
    ln_delta <- (ln_delta_num - p0) / ln_delta_den
    all_sev <- unlist(sev_list[policy_idx])
    lx_gamma <- mean(exp(-kernel_gamma * all_sev))
  }
  
  psi_fun <- switch(
    kernel_type,
    linear = function(n, mu)     (n - mu) / sqrt(mu),
    exponential    = function(n, mu)     exp(-kernel_gamma * n) - ln_delta
  )
  phi_fun <- switch(
    kernel_type,
    linear = function(x, a, b)   (x - a*b) / sqrt(a*b),
    exponential    = function(x, a, b)   exp(-kernel_gamma * x) - lx_gamma
  )
  
  psi_pol <- psi_fun(n_pol, E_N)
  psi_all2_pol <- np$array(as.numeric(psi_pol), dtype='float32')
  y_sev_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
  
  # --- Alternating NN and Omega IFM Loop (severity unchanged)
  max_iter <- 50
  tol <- 1e-1
  omega_nn <- 0.1
  omega_trace <- numeric(max_iter+1)
  omega_trace[1] <- omega_nn
  for(loop in 1:max_iter) {
    sev_loss_with_omega <- if(sev_dist=="Gamma") {
      function(y_true, y_pred) {
        sev_mean_scaled  <- y_true[,1]
        psi       <- y_true[,2]
        beta_scaled      <- y_pred[,1]
        sev_mean <- sev_mean_scaled * sev_mean_sd + sev_mean_mean
        beta     <- beta_scaled * sev_mean_sd + sev_mean_mean
        E_X <- alpha_hat * beta
        Var_X <- alpha_hat * (beta^2)
        phi_val <- if (kernel_type == "standardized") (sev_mean - E_X) / sqrt(Var_X + 1e-8)
        else exp(-kernel_gamma * sev_mean) - lx_gamma
        sarmanov_weight <- 1 + omega_nn * psi * phi_val
        sarmanov_weight <- k_maximum(sarmanov_weight, 1e-8)
        ll <- (alpha_hat-1)*k_log(sev_mean + 1e-8) - sev_mean/beta -
          alpha_hat*k_log(beta) - tf$math$lgamma(tf$cast(alpha_hat, tf$float32))
        loss <- -(ll + k_log(sarmanov_weight))
        return(loss)
      }
    } else {
      function(y_true, y_pred) {
        sev_mean_scaled  <- y_true[,1]
        psi       <- y_true[,2]
        mu_log_scaled    <- y_pred[,1]
        sev_mean <- sev_mean_scaled * sev_mean_sd + sev_mean_mean
        mu_log   <- mu_log_scaled * sev_mean_sd + sev_mean_mean
        E_X <- exp(mu_log + 0.5 * sigma_hat^2)
        Var_X <- (exp(sigma_hat^2) - 1) * exp(2 * mu_log + sigma_hat^2)
        phi_val <- if (kernel_type == "standardized") (sev_mean - E_X) / sqrt(Var_X + 1e-8)
        else exp(-kernel_gamma * sev_mean) - lx_gamma
        sarmanov_weight <- 1 + omega_nn * psi * phi_val
        sarmanov_weight <- k_maximum(sarmanov_weight, 1e-8)
        ll <- -0.5*log(2*pi) - log(sigma_hat) - 0.5 * ((log(sev_mean + 1e-8) - mu_log)/sigma_hat)^2 - log(sev_mean + 1e-8)
        loss <- -(ll + k_log(sarmanov_weight))
        return(loss)
      }
    }
    keras::k_clear_session()
    input_g <- layer_input(shape=p, name="sev_cov")
    out_g <- input_g
    for (i in seq_along(hyper$sev_units)) {
      out_g <- out_g %>%
        layer_dense(units=hyper$sev_units[i], activation=hyper$sev_activations[i])
      if (hyper$sev_dropout[i] > 0)
        out_g <- out_g %>% layer_dropout(rate=hyper$sev_dropout[i])
    }
    out_g <- out_g %>%
      layer_dense(units=1, activation="linear",
                  name=if(sev_dist=="Gamma") "beta_scaled" else "mu_log_scaled")
    sev_model <- keras_model(inputs=input_g, outputs=out_g)
    sev_model$compile(optimizer = optimizer_adam(learning_rate = hyper$lr_sev),
                      loss = sev_loss_with_omega)
    sev_model$fit(
      x = xg_pol, y = y_sev_pol,
      epochs = as.integer(hyper$epochs), batch_size = as.integer(hyper$batch_size),
      verbose = verbose
    )
    # Update predictions and psi/phi
    freq_pred_pol_scaled <- as.vector(freq_model$predict(
      np$array(matrix(as.numeric(X_pol_scaled), ncol=p), dtype="float32")))
    
    if (freq_dist == "Poisson") {
      lambda_pred_pol <- freq_pred_pol_scaled * freq_sd + freq_mean
      freq_pred_pol <- lambda_pred_pol
      E_N <- freq_pred_pol
      Var_N <- freq_pred_pol
    } else if (freq_dist == "ZIP") {
      lambda_pred_pol <- freq_pred_pol_scaled * freq_sd + freq_mean
      freq_pred_pol <- (1 - pi_hat) * lambda_pred_pol
      E_N <- freq_pred_pol
      Var_N <- (1 - pi_hat) * lambda_pred_pol * (1 + pi_hat * lambda_pred_pol)
    } else if (freq_dist == "NB") {
      p_pred_pol <- freq_pred_pol_scaled * p_sd + p_mean
      p_pred_pol <- pmin(pmax(p_pred_pol, 1e-8), 1-1e-8)
      freq_pred_pol <- r_hat * (1 - p_pred_pol) / p_pred_pol
      E_N <- freq_pred_pol
      Var_N <- freq_pred_pol + (freq_pred_pol^2) / r_hat
    }
    psi_pol <- psi_fun(n_pol, E_N)
    psi_all2_pol <- np$array(as.numeric(psi_pol), dtype='float32')
    y_sev_pol <- np$column_stack(list(yg_pol, psi_all2_pol))
    
    # Omega re-estimation
    phi_pol <- if(sev_dist=="Gamma") {
      beta_pred_pol_scaled <- as.vector(sev_model$predict(
        np$array(matrix(as.numeric(X_pol_scaled), ncol=p), dtype="float32")))
      beta_pred_pol <- beta_pred_pol_scaled * sev_mean_sd + sev_mean_mean
      E_X <- alpha_hat * beta_pred_pol
      Var_X <- alpha_hat * (beta_pred_pol^2)
      if (kernel_type == "standardized") (sev_mean - E_X) / sqrt(Var_X + 1e-8)
      else exp(-kernel_gamma * sev_mean) - lx_gamma
    } else {
      mu_log_pred_pol_scaled <- as.vector(sev_model$predict(
        np$array(matrix(as.numeric(X_pol_scaled), ncol=p), dtype="float32")))
      mu_log_pred_pol <- mu_log_pred_pol_scaled * sev_mean_sd + sev_mean_mean
      E_X <- exp(mu_log_pred_pol + 0.5 * sigma_hat^2)
      Var_X <- (exp(sigma_hat^2) - 1) * exp(2 * mu_log_pred_pol + sigma_hat^2)
      if (kernel_type == "linear") (sev_mean - E_X) / sqrt(Var_X + 1e-8)
      else exp(-kernel_gamma * sev_mean) - lx_gamma
    }
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
  }
  
  # ---- Frequency RMSE on Validation ----
  freq_pred_val_scaled <- as.vector(freq_model$predict(x_val))
  if (freq_dist == "Poisson") {
    freq_pred_val <- freq_pred_val_scaled * freq_sd + freq_mean
    freq_pred_mean <- freq_pred_val
  } else if (freq_dist == "ZIP") {
    lambda_pred_val <- freq_pred_val_scaled * freq_sd + freq_mean
    freq_pred_mean <- (1 - pi_hat) * lambda_pred_val
  } else if (freq_dist == "NB") {
    p_pred_val <- freq_pred_val_scaled * p_sd + p_mean
    p_pred_val <- pmin(pmax(p_pred_val, 1e-8), 1-1e-8)
    freq_pred_mean <- r_hat * (1 - p_pred_val) / p_pred_val
  }
  rmse_freq <- sqrt(mean((as.numeric(freq[val_idx]) - freq_pred_mean)^2))
  
  # ---- Severity RMSE on Policies ----
  sev_pred_val_scaled <- as.vector(sev_model$predict(xg_pol))
  sev_pred_val <- sev_pred_val_scaled * sev_mean_sd + sev_mean_mean
  if(sev_dist == "Gamma") {
    sev_pred_mean <- alpha_hat * sev_pred_val
    sev_obs <- sapply(sev_list[policy_idx], mean)
  } else {
    mu_log_pred_val <- sev_pred_val
    sev_pred_mean <- exp(mu_log_pred_val + 0.5 * sigma_hat^2)
    sev_obs <- sapply(sev_list[policy_idx], mean)
  }
  rmse_sev <- sqrt(mean((sev_obs - sev_pred_mean)^2))
  
  # ---- Kolmogorov–Smirnov statistic for severity ----
  if(sev_dist == "Gamma") {
    ks_stat <- suppressWarnings(ks.test(unlist(sev_list), "pgamma",
                                        shape = alpha_hat, scale = mean(sev_pred_val)))$statistic
  } else {
    ks_stat <- suppressWarnings(ks.test(unlist(sev_list), "plnorm",
                                        meanlog = mean(mu_log_pred_val), sdlog = sigma_hat))$statistic
  }
  
  # ---- Empirical Correlation ----
  empirical_corr <- tryCatch({
    cor(freq[freq > 0], sapply(sev_list[freq > 0], mean))
  }, error = function(e) NA)
  
  # ---- Log-likelihood for Sarmanov Model ----
  nll <- 0
  for(i in seq_along(policy_idx)) {
    ni <- n_pol[i]
    # Frequency
    if(freq_dist == "Poisson") {
      ll_freq <- dpois(ni, lambda = E_N[i], log=TRUE)
    } else if(freq_dist == "ZIP") {
      ll_freq <- if(ni == 0) {
        log(pi_hat + (1 - pi_hat) * dpois(0, lambda = lambda_pred_pol[i]))
      } else {
        log(1 - pi_hat) + dpois(ni, lambda = lambda_pred_pol[i], log=TRUE)
      }
    } else if(freq_dist == "NB") {
      ll_freq <- dnbinom(ni, size = r_hat, mu = E_N[i], log=TRUE)
    }
    # Severity (mean)
    sev_i <- sev_mean[i]
    if(sev_dist == "Gamma") {
      ll_sev <- dgamma(sev_i, shape=alpha_hat, scale=sev_pred_val[i], log=TRUE)
      E_Xi <- alpha_hat * sev_pred_val[i]
      Var_Xi <- alpha_hat * sev_pred_val[i]^2
    } else {
      ll_sev <- dlnorm(sev_i, meanlog=sev_pred_val[i], sdlog=sigma_hat, log=TRUE)
      E_Xi <- exp(sev_pred_val[i] + 0.5 * sigma_hat^2)
      Var_Xi <- (exp(sigma_hat^2) - 1) * exp(2 * sev_pred_val[i] + sigma_hat^2)
    }
    psi_i <- psi_fun(ni, E_N[i])
    phi_i <- if(sev_dist=="Gamma") {
      if (kernel_type == "standardized") (sev_i - E_Xi) / sqrt(Var_Xi + 1e-8)
      else exp(-kernel_gamma * sev_i) - lx_gamma
    } else {
      if (kernel_type == "standardized") (sev_i - E_Xi) / sqrt(Var_Xi + 1e-8)
      else exp(-kernel_gamma * sev_i) - lx_gamma
    }
    W <- 1 + omega_nn * psi_i * phi_i
    if(!is.finite(W) || W <= 0) W <- 1e-8
    nll <- nll + (ll_freq + ll_sev + log(W))
  }
  LL <- nll
  n_par <- 1 # omega
  if(freq_dist == "ZIP") n_par <- n_par + 1
  if(freq_dist == "NB") n_par <- n_par + 1
  if(sev_dist == "Gamma") n_par <- n_par + 1
  if(sev_dist == "Lognormal") n_par <- n_par + 1
  n_par <- n_par +
    sum(sapply(freq_model$weights, function(w) prod(dim(w)))) +
    sum(sapply(sev_model$weights, function(w) prod(dim(w))))
  AIC <- 2*n_par - 2*LL
  BIC <- log(length(policy_idx))*n_par - 2*LL
  
  # --- Chi-squared for frequency
  max_count <- max(freq)
  obs_tab <- tabulate(freq + 1, nbins = max_count + 1)
  exp_tab <- numeric(max_count + 1)
  for (k in 0:max_count) {
    if (freq_dist == "Poisson") {
      exp_tab[k + 1] <- sum(dpois(k, lambda = E_N))
    } else if (freq_dist == "ZIP") {
      exp_tab[k + 1] <- sum(
        if (k == 0) pi_hat + (1 - pi_hat) * dpois(0, lambda = lambda_pred_pol)
        else (1 - pi_hat) * dpois(k, lambda = lambda_pred_pol)
      )
    } else if (freq_dist == "NB") {
      exp_tab[k + 1] <- sum(dnbinom(k, size = r_hat, mu = E_N))
    }
  }
  chi2_freq <- sum((obs_tab - exp_tab)^2 / (exp_tab + 1e-8))
  
  # --- Chi-squared for severity (on mean severities)
  sev_bins <- quantile(unlist(sev_list), probs=seq(0,1,length=11)) # deciles
  obs_sev_tab <- hist(sev_mean, breaks=sev_bins, plot=FALSE)$counts
  exp_sev_tab <- numeric(length(obs_sev_tab))
  if(sev_dist == "Gamma") {
    for(i in seq_along(exp_sev_tab)) {
      a <- sev_bins[i]; b <- sev_bins[i+1]
      exp_sev_tab[i] <- sum(pgamma(b, shape=alpha_hat, scale=sev_pred_val) -
                              pgamma(a, shape=alpha_hat, scale=sev_pred_val))
    }
  } else {
    for(i in seq_along(exp_sev_tab)) {
      a <- sev_bins[i]; b <- sev_bins[i+1]
      exp_sev_tab[i] <- sum(plnorm(b, meanlog=sev_pred_val, sdlog=sigma_hat) -
                              plnorm(a, meanlog=sev_pred_val, sdlog=sigma_hat))
    }
  }
  exp_sev_tab <- exp_sev_tab * length(sev_mean)
  chi2_sev <- sum((obs_sev_tab - exp_sev_tab)^2 / (exp_sev_tab + 1e-8))
  
  summary_tab <- tibble(
    Model = paste(freq_dist, sev_dist, kernel_type, sep="-"),
    LL    = LL,
    AIC   = AIC,
    BIC   = BIC,
    RMSE_freq = rmse_freq,
    RMSE_sev  = rmse_sev,
    Chi2_freq = chi2_freq,
    Chi2_sev  = chi2_sev,
    Omega     = omega_nn,
    KS_sev    = as.numeric(ks_stat),
    EmpCorr   = empirical_corr
  )
  print(kable(summary_tab, digits=4))
  
  return(list(
    freq_model = freq_model,
    sev_model  = sev_model,
    omega_nn   = omega_nn,
    freq_dist  = freq_dist,
    sev_dist   = sev_dist,
    kernel_type = kernel_type,
    kernel_gamma = kernel_gamma,
    pi_hat     = if(exists("pi_hat")) pi_hat else NULL,
    r_hat      = if(exists("r_hat")) r_hat else NULL,
    alpha_hat  = if(exists("alpha_hat")) alpha_hat else NULL,
    sigma_hat  = if(exists("sigma_hat")) sigma_hat else NULL,
    scaler = list(
      X_mean=X_mean, X_sd=X_sd, freq_mean=freq_mean, freq_sd=freq_sd,
      sev_mean_mean=sev_mean_mean, sev_mean_sd=sev_mean_sd, X_pol_mean=X_pol_mean, X_pol_sd=X_pol_sd,
      p_mean=if(exists("p_mean")) p_mean else NULL,
      p_sd=if(exists("p_sd")) p_sd else NULL
    ),
    hyper = hyper,
    summary_tab = summary_tab
  ))
}



library(ggplot2)
library(knitr)

compare_aggregate_losses <- function(
    fit,                # fitted Sarmanov-NN object
    X,                  # full covariate matrix (n x p)
    freq,               # frequency vector (n)
    sev_list,           # list of severities
    val_idx,            # validation indices (vector)
    Msim = 10000,       # number of simulation replicates
    quantiles = c(0.25, 0.5, 0.75, 0.95, 0.99, 0.999),
    seed = 369,
    plot = TRUE
) {
  set.seed(seed)
  # --- Extract scaling and fitted info ---
  X <- as.matrix(X)
  scaler <- fit$scaler
  freq_model <- fit$freq_model
  sev_model  <- fit$sev_model
  freq_dist <- fit$freq_dist
  sev_dist  <- fit$sev_dist
  pi_hat    <- fit$pi_hat
  r_hat     <- fit$r_hat
  alpha_hat <- fit$alpha_hat
  sigma_hat <- fit$sigma_hat
  p_mean    <- scaler$p_mean
  p_sd      <- scaler$p_sd
  freq_mean <- scaler$freq_mean
  freq_sd   <- scaler$freq_sd
  sev_mean_mean <- scaler$sev_mean_mean
  sev_mean_sd   <- scaler$sev_mean_sd
  X_mean <- scaler$X_mean
  X_sd   <- scaler$X_sd
  X_pol_mean <- scaler$X_pol_mean
  X_pol_sd   <- scaler$X_pol_sd
  
  val_X <- X[val_idx, , drop = FALSE]
  n_val <- length(val_idx)
  
  # --- Predicted frequency/parameter ---
  val_X_scaled <- scale(val_X, center = X_mean, scale = X_sd)
  freq_pred_scaled <- as.vector(freq_model$predict(
    np$array(matrix(as.numeric(val_X_scaled), ncol = ncol(val_X)), dtype = "float32")
  ))
  if (freq_dist == "Poisson") {
    freq_pred <- freq_pred_scaled * freq_sd + freq_mean
  } else if (freq_dist == "ZIP") {
    lambda_pred <- freq_pred_scaled * freq_sd + freq_mean
    freq_pred <- (1 - pi_hat) * lambda_pred
  } else if (freq_dist == "NB") {
    p_pred <- freq_pred_scaled * p_sd + p_mean
    p_pred <- pmin(pmax(p_pred, 1e-8), 1 - 1e-8)
    freq_pred <- r_hat * (1 - p_pred) / p_pred
  }
  
  # --- Predicted severity/parameter ---
  pol_idx <- which(freq[val_idx] > 0)
  val_X_pol <- val_X[pol_idx, , drop=FALSE]
  val_X_pol_scaled <- scale(val_X_pol, center = X_pol_mean, scale = X_pol_sd)
  sev_pred_scaled <- as.vector(sev_model$predict(
    np$array(matrix(as.numeric(val_X_pol_scaled), ncol=ncol(val_X)), dtype="float32")
  ))
  if (sev_dist == "Gamma") {
    sev_pred <- sev_pred_scaled * sev_mean_sd + sev_mean_mean
  } else if (sev_dist == "Lognormal") {
    sev_pred <- sev_pred_scaled * sev_mean_sd + sev_mean_mean
  }
  
  # --- Simulate aggregate losses: predicted vs true ---
  S_pred <- numeric(Msim)
  S_true <- numeric(Msim)
  for (m in seq_len(Msim)) {
    # Simulate one aggregate loss from prediction
    j <- sample(seq_along(val_idx), 1)
    n_pred <- 0
    if (freq_dist == "Poisson") {
      n_pred <- rpois(1, freq_pred[j])
    } else if (freq_dist == "ZIP") {
      n_pred <- if (runif(1) < pi_hat) 0 else rpois(1, lambda_pred[j])
    } else if (freq_dist == "NB") {
      mu_j <- freq_pred[j]
      n_pred <- rnbinom(1, size = r_hat, mu = mu_j)
    }
    if (n_pred == 0) {
      S_pred[m] <- 0
    } else {
      if (sev_dist == "Gamma") {
        # Defensive: check for valid index/NA
        idx <- which(pol_idx == j)
        if (length(idx) > 0 && !is.na(sev_pred[idx]) && sev_pred[idx] > 0 && is.finite(sev_pred[idx])) {
          S_pred[m] <- sum(rgamma(n_pred, shape = alpha_hat, scale = sev_pred[idx]))
        } else {
          S_pred[m] <- NA_real_
        }
      } else if (sev_dist == "Lognormal") {
        idx <- which(pol_idx == j)
        if (length(idx) > 0 && !is.na(sev_pred[idx]) && is.finite(sev_pred[idx])) {
          S_pred[m] <- sum(rlnorm(n_pred, meanlog = sev_pred[idx], sdlog = sigma_hat))
        } else {
          S_pred[m] <- NA_real_
        }
      } else {
        S_pred[m] <- NA_real_
      }
    }
    # Simulate one aggregate loss from *true* data (validation)
    j_true <- sample(seq_along(val_idx), 1)
    S_true[m] <- sum(sev_list[[val_idx[j_true]]], na.rm=TRUE)
    if (!is.finite(S_true[m])) S_true[m] <- NA_real_
  }
  
  # --- Remove NA/NaN/Inf ---
  S_pred <- S_pred[is.finite(S_pred) & !is.na(S_pred) & !is.nan(S_pred)]
  S_true <- S_true[is.finite(S_true) & !is.na(S_true) & !is.nan(S_true)]
  
  # --- Summarize ---
  summarize_vec <- function(x) {
    c(Mean = mean(x, na.rm=TRUE), quantile(x, probs = quantiles, na.rm=TRUE))
  }
  summary_pred <- summarize_vec(S_pred)
  summary_true <- summarize_vec(S_true)
  agg_tab <- rbind(Predicted = summary_pred, True = summary_true)
  colnames(agg_tab) <- c("Mean", paste0(names(quantile(S_pred, probs=quantiles, na.rm=TRUE))))
  
  print(kable(agg_tab, digits=4, caption = "Aggregate-loss summaries: Predicted vs True (Validation Set)"))
  
  # --- Optional plot ---
  if (plot) {
    n_pred <- length(S_pred)
    n_true <- length(S_true)
    df <- data.frame(
      agg_loss = c(S_pred, S_true),
      Type = rep(c("Predicted", "True"), c(n_pred, n_true))
    )
    p <- ggplot(df, aes(x = agg_loss, color = Type, fill=Type)) +
      geom_density(alpha = 0.2, lwd=1.1, na.rm=TRUE) +
      labs(title = "Aggregate Loss Distribution (Validation Set)",
           x = "Aggregate Loss", y = "Density") +
      theme_minimal()
    print(p)
  }
  
  invisible(list(
    predicted = S_pred,
    true = S_true,
    agg_tab = agg_tab
  ))
}


###########example use -----
# --------Sample fitting --------
# Prepare covariate matrix (n x p)
X <- matrix(age, ncol=1)   # or: data.frame(age=age)
colnames(X) <- "age"


hyper = list(
  freq_units = c(128, 64, 32),
  freq_activations = c("relu", "relu", "relu"),
  freq_dropout = c(0.05, 0.05, 0.05),
  sev_units = c(128, 64, 32, 16, 8),
  sev_activations = rep("relu", 5),
  sev_dropout = rep(0.05, 5),
  epochs = 150,
  batch_size = 128,
  lr_freq = 6e-3,
  lr_sev = 1e-3
)


fit <- fit_sarmanov_nn(
  freq_dist = "Poisson",   # or "NB", "ZIP"
  sev_dist  = "Gamma",  # or "Lognormal"
  X         = X,  # covariates
  kernel_type = "standardized", # or "standardized",
  kernel_gamma = 1,
  freq      = freq,
  sev_list  = sev_list,
  hyper = list(
    freq_units = c(128, 64, 32),
    freq_activations = c("relu", "relu", "relu"),
    freq_dropout = c(0.05, 0.05, 0.05),
    sev_units = c(128, 64, 32, 16, 8),
    sev_activations = rep("relu", 5),
    sev_dropout = rep(0.05, 5),
    epochs = 150,
    batch_size = 128,
    lr_freq = 6e-3,
    lr_sev = 1e-3
  )
  
  # ...hyperparameters for NN... e.g.hyper = list(
  #freq_units = c(128,32),
  #freq_activations = c("relu", "relu"),
  #freq_dropout = c(0.1, 0.1),
  #sev_units = c(64,32,16,8),
  #sev_activations = c("relu","relu","relu","relu"),
  #sev_dropout = c(0.1,0.1,0.05,0.05),
  #epochs = 150,
  #batch_size = 256,
  #lr_freq = 6e-3,
  #lr_sev = 1e-3
  #),
  #verbose=2
)

# ------plot

age_grid <- seq(min(X[,1]), max(X[,1]), length.out = 100)  # or whatever your covariate is named
X_grid <- matrix(age_grid, ncol=1)

colnames(X_grid) <- colnames(X)

X_grid_scaled <- scale(X_grid, center = fit$scaler$X_mean, scale = fit$scaler$X_sd)


lambda_pred_scaled <- as.vector(
  fit$freq_model$predict(np$array(X_grid_scaled, dtype = "float32"))
)
lambda_pred <- lambda_pred_scaled * fit$scaler$freq_sd + fit$scaler$freq_mean


lambda_true <- 0.003 * (age_grid - 30)^2 + exp(0.05*(age_grid - 40) - 0.008*(age_grid - 40)^2 + log(3)) + 0.6

plot(age_grid, lambda_pred, type = "l", col = "purple", lwd = 2,
     ylab = expression(lambda), xlab = "Age",
     main = "Predicted vs True Frequency (Lambda) vs Age")

if (exists("lambda_true")) {
  lines(age_grid, lambda_true, col = "black", lwd = 2, lty = 2)
  legend("topleft",
         legend = c("NN Prediction", "True"),
         col = c("purple", "black"), lwd = 2, lty = c(1,2))
}


beta_pred_scaled <- as.vector(
  fit$sev_model$predict(np$array(X_grid_scaled, dtype = "float32"))
)

beta_pred <- beta_pred_scaled * fit$scaler$sev_mean_sd + fit$scaler$sev_mean_mean

beta_pred <- beta_pred_scaled * fit$scaler$sev_nn_sd + fit$scaler$sev_nn_mean

mu_true     <- 0.08*abs(age_grid-40) + exp(0.3*sqrt(age_grid) - 0.12*(age_grid-40)^2 + log(0.5)) + 2
beta_true   <- mu_true / 2.87


plot(age_grid, beta_pred, type = "l", col = "purple", lwd = 2,
     ylab = expression(beta), xlab = "Age",
     main = "Predicted vs True Frequency (Beta) vs Age", ylim = c(0,3)
)

lines(age_grid, beta_true, col = "black", lwd = 2, lty = 2)

legend("topleft",
         legend = c("NN Prediction", "True"),
         col = c("purple", "black"), lwd = 2, lty = c(1,2))



set.seed(369)
n <- nrow(X)
ix <- sample(n)
val_idx <- as.integer(ix[1:round(n * 0.1)])


compare_aggregate_losses(
  fit     = fit,
  X       = X,
  freq    = freq,
  sev_list = sev_list,
  val_idx = val_idx,
  Msim    = 10000     # Number of simulated samples for robust tail estimation
)
