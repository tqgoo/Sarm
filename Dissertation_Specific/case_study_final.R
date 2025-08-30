# ──────────────────────────────────────────────────────────────────────────────
# Multi-covariate ZIP + Gamma + Sarmanov (IFM with NNs)
# - Train-only pi_hat (global zero rate)
# - Train-only scalers (val scalers saved for reference)
# - NA-safe feature cleaning (numeric median; factors -> "__MISSING__")
# - Stabilized ω update + robust Gamma-shape estimation
# - NEGATIVE λ → CLAMP TO ZERO (three places marked "← CLAMP")
# ──────────────────────────────────────────────────────────────────────────────
fit_zip_gamma_multi <- function(
    X, freq, sev_list,
    kernel = "exponential",   # or "standardized"
    delta  = 0.5,             # exponential psi
    gammaL = 0.5,             # exponential phi
    hyper = list(
      freq_units       = c(128, 32),
      freq_activations = c("relu","relu"),
      freq_dropout     = c(0.15, 0.10),
      sev_units        = c(64, 32, 16, 8),
      sev_activations  = c("relu","relu","relu","relu"),
      sev_dropout      = c(0.15, 0.10, 0.05, 0.05),
      epochs           = 250,
      batch_size       = 256,
      lr_freq          = 6e-3,
      lr_sev           = 1e-3
    ),
    max_iter = 50, tol = 1e-2, verbose = 2
) {
  # --- deps ---
  require(MASS)
  require(keras)
  require(tensorflow)
  require(reticulate)
  np <- reticulate::import("numpy")
  tf <- tensorflow::tf
  
  # ── Preflight checks
  if (!(is.data.frame(X) || is.matrix(X))) stop("X must be a data.frame or matrix.")
  X <- as.data.frame(X)
  if (length(freq) != nrow(X)) stop("length(freq) must equal nrow(X).")
  if (length(sev_list) != nrow(X)) stop("length(sev_list) must equal nrow(X).")
  if (!all(vapply(sev_list, function(v) is.numeric(v) || length(v)==0, TRUE)))
    stop("All elements of sev_list must be numeric vectors (possibly length 0).")
  
  # ── 0) Clean X: keep ALL rows, impute/factorize
  for (nm in names(X)) {
    if (is.numeric(X[[nm]])) {
      X[[nm]][!is.finite(X[[nm]])] <- NA
      med <- suppressWarnings(stats::median(X[[nm]], na.rm = TRUE))
      if (!is.finite(med)) med <- 0
      X[[nm]][is.na(X[[nm]])] <- med
    } else {
      X[[nm]] <- as.character(X[[nm]])
      X[[nm]][is.na(X[[nm]]) | X[[nm]] == ""] <- "__MISSING__"
      X[[nm]] <- factor(X[[nm]])
    }
  }
  
  # Design matrix (no row dropping)
  mm <- model.matrix(~ . - 1, data = X, na.action = na.pass)
  mm[!is.finite(mm)] <- 0
  
  if (nrow(mm) != length(freq)) stop("After model.matrix, nrow(mm) != length(freq).")
  n <- nrow(mm); p <- ncol(mm)
  if (verbose > 0) cat(sprintf("Preflight: n=%d, p=%d, zero rate=%.3f\n",
                               n, p, mean(freq == 0, na.rm = TRUE)))
  
  # ── 1) Train/val split FIRST
  ix     <- sample.int(n)
  n_val  <- max(1L, as.integer(n * 0.1))
  val_id <- ix[1:n_val]
  tr_id  <- ix[(n_val + 1):n]
  if (length(tr_id) < 10) stop("Training split too small after sampling.")
  
  # ── 2) Feature scalers (TRAIN ONLY)
  feat_mean_tr <- colMeans(mm[tr_id, , drop = FALSE])
  feat_sd_tr   <- apply(mm[tr_id, , drop = FALSE], 2, sd)
  feat_sd_tr[feat_sd_tr == 0 | is.na(feat_sd_tr)] <- 1
  
  # Validation scalers (reference)
  feat_mean_val <- colMeans(mm[val_id, , drop = FALSE])
  feat_sd_val   <- apply(mm[val_id, , drop = FALSE], 2, sd)
  feat_sd_val[feat_sd_val == 0 | is.na(feat_sd_val)] <- 1
  
  # Scale ALL rows using TRAIN scalers
  X_scaled <- scale(mm, center = feat_mean_tr, scale = feat_sd_tr)
  
  # ── 3) π_hat from TRAIN ONLY (exact train zero proportion)
  pi_hat <- mean(freq[tr_id] == 0, na.rm = TRUE)
  pi_hat <- min(max(pi_hat, 1e-8), 1 - 1e-8)
  if (verbose > 0) cat(sprintf("Train zero rate (pi_hat) = %.6f\n", pi_hat))
  
  # ── 4) Frequency target (linear λ) and TRAIN scalers
  lambda_zip <- freq / (1 - pi_hat)
  lambda_zip[!is.finite(lambda_zip)] <- 0
  
  freq_mean_tr <- mean(lambda_zip[tr_id])
  freq_sd_tr   <- sd(lambda_zip[tr_id]); if (!is.finite(freq_sd_tr) || freq_sd_tr == 0) freq_sd_tr <- 1
  lambda_zip_scaled <- (lambda_zip - freq_mean_tr) / freq_sd_tr
  
  # Save VAL (reference)
  freq_mean_val <- mean(lambda_zip[val_id])
  freq_sd_val   <- sd(lambda_zip[val_id]); if (!is.finite(freq_sd_val) || freq_sd_val == 0) freq_sd_val <- 1
  
  # ── 5) Build tensors
  x_train <- np$array(X_scaled[tr_id, , drop = FALSE], dtype = "float32")
  y_train <- np$array(as.numeric(lambda_zip_scaled[tr_id]), dtype = "float32")
  x_val   <- np$array(X_scaled[val_id, , drop = FALSE], dtype = "float32")
  y_val   <- np$array(as.numeric(lambda_zip_scaled[val_id]), dtype = "float32")
  
  # ── 6) Frequency NN (Keras 3)
  keras::k_clear_session()
  input_p <- layer_input(shape = p, name = "X_p")
  h <- input_p
  for (i in seq_along(hyper$freq_units)) {
    h <- h %>%
      layer_dense(units = hyper$freq_units[i], activation = hyper$freq_activations[i]) %>%
      layer_dropout(rate = hyper$freq_dropout[i])
  }
  out_p <- h %>% layer_dense(units = 1, activation = "linear", name = "lambda_zip_scaled")
  zip_model <- keras_model(inputs = input_p, outputs = out_p)
  zip_model$compile(optimizer = optimizer_adam(learning_rate = hyper$lr_freq), loss = "mse")
  zip_model$fit(
    x = x_train, y = y_train,
    epochs = as.integer(hyper$epochs), batch_size = as.integer(hyper$batch_size),
    validation_data = list(x_val, y_val), verbose = verbose
  )
  
  # ── 7) Severity aggregates (freq>0)
  policy_idx <- which(freq > 0)
  if (length(policy_idx) == 0) stop("No positive-frequency policies found.")
  X_pol   <- X_scaled[policy_idx, , drop = FALSE]
  n_pol   <- freq[policy_idx]
  
  # Clean sev_list & per-policy means
  sev_list <- lapply(sev_list, function(v) if (length(v)) as.numeric(v) else numeric(0))
  sev_mean <- vapply(sev_list[policy_idx], function(v) if (length(v)) mean(v) else NA_real_, 0.0)
  sev_mean[!is.finite(sev_mean)] <- NA_real_
  
  # Train-only sev scalers (only policies that are in TRAIN)
  pol_tr_id  <- intersect(policy_idx, tr_id)
  pol_val_id <- intersect(policy_idx, val_id)
  
  sev_mean_tr <- vapply(sev_list[pol_tr_id], function(v) if (length(v)) mean(v) else NA_real_, 0.0)
  sev_mean_mean_tr <- mean(sev_mean_tr, na.rm = TRUE)
  sev_mean_sd_tr   <- sd(sev_mean_tr,  na.rm = TRUE); if (!is.finite(sev_mean_sd_tr) || sev_mean_sd_tr == 0) sev_mean_sd_tr <- 1
  sev_mean_scaled  <- (sev_mean - sev_mean_mean_tr) / sev_mean_sd_tr
  
  # Validation sev scalers (reference)
  if (length(pol_val_id) > 0) {
    sev_mean_val <- vapply(sev_list[pol_val_id], function(v) if (length(v)) mean(v) else NA_real_, 0.0)
    sev_mean_mean_val <- mean(sev_mean_val, na.rm = TRUE)
    sev_mean_sd_val   <- sd(sev_mean_val,  na.rm = TRUE); if (!is.finite(sev_mean_sd_val) || sev_mean_sd_val == 0) sev_mean_sd_val <- 1
  } else { sev_mean_mean_val <- NA_real_; sev_mean_sd_val <- NA_real_ }
  
  xg_pol <- np$array(X_pol, dtype = "float32")
  yg_pol <- np$array(as.numeric(sev_mean_scaled), dtype = "float32")
  
  # ── Initial ψ using TRAIN-scaled λ predictions
  lambda_pred_pol_scaled <- as.vector(zip_model$predict(xg_pol, verbose = 0))
  lambda_pred_pol <- lambda_pred_pol_scaled * freq_sd_tr + freq_mean_tr
  lambda_pred_pol_adj <- (1 - pi_hat) * lambda_pred_pol
  lambda_pred_pol_adj <- pmax(lambda_pred_pol_adj, 0)   # ← CLAMP
  
  # ψ / φ helpers
  get_psi_vector <- function(n, lambda, kernel, delta) {
    if (kernel == "exponential") {
      exp(-delta * n) - exp(-delta * lambda)
    } else {
      (n - lambda) / sqrt(pmax(lambda, 1e-8))  # sqrt safety
    }
  }
  psi_pol <- get_psi_vector(n_pol, lambda_pred_pol_adj, kernel, delta)
  psi_np  <- np$array(as.numeric(psi_pol), dtype = "float32")
  y_gamma_pol <- np$column_stack(list(yg_pol, psi_np))
  
  # ── Robust Gamma shape α (winsorized + bounded + MoM fallback)
  sev_all <- unlist(sev_list, use.names = FALSE)
  sev_all <- sev_all[is.finite(sev_all) & sev_all > 0]
  if (length(sev_all) < 2) stop("Not enough positive severities to estimate Gamma shape.")
  q_lo <- as.numeric(stats::quantile(sev_all, 0.001, names = FALSE, type = 7))
  q_hi <- as.numeric(stats::quantile(sev_all, 0.999, names = FALSE, type = 7))
  sev_all_w <- pmin(pmax(sev_all, q_lo), q_hi)
  sev_all_w <- pmax(sev_all_w, 1e-8)
  
  alpha_hat <- tryCatch({
    fit <- MASS::fitdistr(
      x = sev_all_w, densfun = "gamma",
      start = list(shape = 2, rate = 1 / stats::median(sev_all_w)),
      method = "L-BFGS-B", lower = c(shape = 1e-6, rate = 1e-6)
    )
    unname(fit$estimate["shape"])
  }, error = function(e) NA_real_)
  if (!is.finite(alpha_hat) || alpha_hat <= 0) {
    m <- mean(sev_all_w); v <- stats::var(sev_all_w)
    alpha_mom <- (m * m) / max(v, 1e-12)
    alpha_hat <- if (is.finite(alpha_mom) && alpha_mom > 0) alpha_mom else 2.0
  }
  
  # φ helpers
  phi_exp_gamma_tf <- function(x, alpha, beta, gammaL)
    tf$math$exp(-gammaL * x) - tf$math$pow(1 + beta * gammaL, -alpha)
  phi_std_gamma_tf <- function(x, mu, sx) (x - mu) / sx
  get_phi_vector <- function(x, alpha, beta, kernel, gammaL) {
    if (kernel == "exponential") exp(-gammaL * x) - (1 + beta * gammaL)^(-alpha)
    else (x - alpha * beta) / (sqrt(alpha) * beta)
  }
  
  # ── 9) IFM loop (Gamma NN + ω)
  omega_nn <- 0.1
  omega_trace <- numeric(max_iter + 1); omega_trace[1] <- omega_nn
  last_iter <- 0
  
  for (loop in 1:max_iter) {
    if (verbose > 0) cat(sprintf("\n[IFM] Iteration %d — omega=%.5f\n", loop, omega_nn))
    
    gamma_loss_with_omega <- function(y_true, y_pred) {
      sev_mean_scaled_tf <- y_true[, 1]
      psi_tf             <- y_true[, 2]
      beta_scaled_tf     <- y_pred[, 1]
      
      sev_mean_tf <- sev_mean_scaled_tf * sev_mean_sd_tr + sev_mean_mean_tr
      beta_tf     <- beta_scaled_tf     * sev_mean_sd_tr + sev_mean_mean_tr
      beta_tf     <- tf$math$maximum(beta_tf, 1e-6)
      
      if (kernel == "exponential") {
        phi_val <- phi_exp_gamma_tf(sev_mean_tf, alpha_hat, beta_tf, gammaL)
      } else {
        mu_tf <- alpha_hat * beta_tf
        sx_tf <- tf$math$sqrt(alpha_hat) * beta_tf
        phi_val <- phi_std_gamma_tf(sev_mean_tf, mu_tf, sx_tf)
      }
      sarmanov_weight <- 1 + omega_nn * psi_tf * phi_val
      sarmanov_weight <- tf$math$maximum(sarmanov_weight, 1e-8)
      
      ll <- (alpha_hat - 1) * tf$math$log(sev_mean_tf + 1e-8) -
        sev_mean_tf / beta_tf - alpha_hat * tf$math$log(beta_tf + 1e-8)
      -(ll + tf$math$log(sarmanov_weight))
    }
    
    keras::k_clear_session()
    input_g <- layer_input(shape = p, name = "X_g")
    h <- input_g
    for (i in seq_along(hyper$sev_units)) {
      h <- h %>%
        layer_dense(units = hyper$sev_units[i], activation = hyper$sev_activations[i]) %>%
        layer_dropout(rate = hyper$sev_dropout[i])
    }
    out_g <- h %>% layer_dense(units = 1, activation = "linear", name = "beta_scaled")
    gamma_model <- keras_model(inputs = input_g, outputs = out_g)
    gamma_model$compile(optimizer = optimizer_adam(learning_rate = hyper$lr_sev),
                        loss = gamma_loss_with_omega)
    gamma_model$fit(
      x = np$array(X_pol, dtype="float32"), y = y_gamma_pol,
      epochs = as.integer(hyper$epochs), batch_size = as.integer(hyper$batch_size),
      verbose = verbose
    )
    
    # ω step: re-predict λ (TRAIN-scaled), then clamp negatives to zero
    lambda_pred_pol_scaled <- as.vector(zip_model$predict(np$array(X_pol, dtype="float32"), verbose = 0))
    lambda_pred_pol <- lambda_pred_pol_scaled * freq_sd_tr + freq_mean_tr
    lambda_pred_pol_adj <- (1 - pi_hat) * lambda_pred_pol
    lambda_pred_pol_adj <- pmax(lambda_pred_pol_adj, 0)   # ← CLAMP
    
    beta_pred_pol_scaled <- as.vector(gamma_model$predict(np$array(X_pol, dtype="float32"), verbose = 0))
    beta_pred_pol <- beta_pred_pol_scaled * sev_mean_sd_tr + sev_mean_mean_tr
    beta_pred_pol[beta_pred_pol <= 1e-8 | !is.finite(beta_pred_pol)] <- 1e-6
    
    psi_pol <- get_psi_vector(n_pol, lambda_pred_pol_adj, kernel, delta)
    phi_pol <- get_phi_vector(sev_mean, alpha_hat, beta_pred_pol, kernel, gammaL)
    
    # Stabilized ω update
    z <- psi_pol * phi_pol
    finite <- is.finite(z) & (z != 0); z <- z[finite]
    
    if (length(z) == 0) {
      omega_new <- 0; lower <- -1; upper <- 1
      if (verbose > 0) cat("  → All z zero/non-finite; set omega = 0\n")
    } else {
      zmax <- max(abs(z))
      omega_cap <- 0.95 / zmax
      lower_feas <- if (any(z > 0)) max(-1 / z[z > 0]) else -Inf
      upper_feas <- if (any(z < 0)) min(-1 / z[z < 0]) else  Inf
      lower <- max(if (is.finite(lower_feas)) lower_feas else -Inf, -omega_cap)
      upper <- min(if (is.finite(upper_feas)) upper_feas else  Inf,  omega_cap)
      eps <- 1e-9
      if (!is.finite(lower)) lower <- -omega_cap
      if (!is.finite(upper)) upper <-  omega_cap
      lower <- lower + eps; upper <- upper - eps
      if (!(lower < upper)) { lower <- -0.5 * omega_cap; upper <- 0.5 * omega_cap }
      
      lambda_pen <- 1e-4
      score      <- function(w) { -sum(z / (1 + w * z)) + 2 * lambda_pen * w }
      negll_omega<- function(w) {
        W <- 1 + w * (psi_pol * phi_pol)
        if (any(!is.finite(W)) || any(W <= 0)) return(Inf)
        -sum(log(W)) + lambda_pen * w^2
      }
      ur <- try(uniroot(score, lower = lower, upper = upper, tol = 1e-9), silent = TRUE)
      if (!inherits(ur, "try-error") && is.finite(ur$root)) omega_new <- ur$root
      else omega_new <- optimize(negll_omega, interval = c(lower, upper))$minimum
    }
    
    if (verbose > 0) cat("  → Estimated omega (stabilized):", signif(omega_new,6),
                         " | bounds [", signif(lower,6), ",", signif(upper,6), "]\n")
    
    omega_trace[loop + 1] <- omega_new
    last_iter <- loop
    if (abs(omega_new - omega_nn) < tol) {
      if (verbose > 0) cat("Converged. Stopping IFM loop.\n")
      omega_nn <- omega_new
      break
    }
    omega_nn <- omega_new
    
    # refresh payload (pattern)
    psi_pol_new <- get_psi_vector(n_pol, lambda_pred_pol_adj, kernel, delta)
    y_gamma_pol <- np$column_stack(list(
      np$array(as.numeric(sev_mean_scaled), dtype = "float32"),
      np$array(as.numeric(psi_pol_new), dtype = "float32")
    ))
  }
  
  # ── 10) Final predictions (TRAIN scalers), clamp negatives to zero
  x_all <- np$array(X_scaled, dtype = "float32")
  lambda_pred_scaled_all <- as.vector(zip_model$predict(x_all, verbose = 0))
  lambda_pred_all <- (lambda_pred_scaled_all * freq_sd_tr + freq_mean_tr)
  lambda_pred_all_adj <- (1 - pi_hat) * lambda_pred_all
  lambda_pred_all_adj <- pmax(lambda_pred_all_adj, 0)   # ← CLAMP
  
  beta_pred_scaled_all <- as.vector(gamma_model$predict(x_all, verbose = 0))
  beta_pred_all <- beta_pred_scaled_all * sev_mean_sd_tr + sev_mean_mean_tr
  
  list(
    zip_model   = zip_model,
    gamma_model = gamma_model,
    omega_nn    = omega_nn,
    omega_trace = omega_trace[1:(last_iter + 1)],
    alpha_hat   = alpha_hat,
    pi_hat      = pi_hat,  # train-only global π
    indices = list(tr_id = tr_id, val_id = val_id, policy_idx = policy_idx,
                   pol_tr_id = pol_tr_id, pol_val_id = pol_val_id),
    internals = list(
      X_scaled = X_scaled,
      sev_mean = sev_mean
    ),
    scaler_train = list(
      feature_names = colnames(mm),
      feat_mean = feat_mean_tr, feat_sd = feat_sd_tr,
      freq_mean = freq_mean_tr, freq_sd = freq_sd_tr,
      sev_mean_mean = sev_mean_mean_tr, sev_mean_sd = sev_mean_sd_tr
    ),
    scaler_val = list(
      feat_mean = feat_mean_val, feat_sd = feat_sd_val,
      freq_mean = freq_mean_val, freq_sd = freq_sd_val,
      sev_mean_mean = sev_mean_mean_val, sev_mean_sd = sev_mean_sd_val
    ),
    preds = list(
      lambda = lambda_pred_all_adj,  # (1-π) * λ(x), negatives → 0
      beta   = beta_pred_all         # Gamma scale; mean severity = α * β(x)
    ),
    kernel = kernel, delta = delta, gammaL = gammaL
  )
}


# ==== Apply fit_zip_gamma_multi() to French MTPL ============================
suppressPackageStartupMessages({
  library(CASdatasets)
  library(dplyr)
  library(ggplot2)
})

# 0) Load MTPL
data(freMTPL2freq)
data(freMTPL2sev)
freq_df <- as.data.frame(freMTPL2freq)
sev_df  <- as.data.frame(freMTPL2sev)

# 1) Build per-policy sev_list from claim-level table
sev_pos <- sev_df %>%
  filter(is.finite(ClaimAmount), ClaimAmount > 0) %>%
  group_by(IDpol) %>%
  summarise(sev_list = list(as.numeric(ClaimAmount)), .groups = "drop")

# 2) Policy table (keep Exposure > 0 to avoid degenerate rows)
dat_policy <- freq_df %>%
  filter(Exposure > 0) %>%
  left_join(sev_pos, by = "IDpol") %>%
  mutate(
    sev_list = lapply(sev_list, function(x) if (is.null(x)) numeric(0) else x),
    freq     = as.numeric(ClaimNb)
  )

# 3) Choose covariates + add logExposure (helps the frequency net learn the offset)
covars <- intersect(
  c("DrivAge","VehAge","VehPower","VehBrand","VehGas","Area","Region","Density","BonusMalus"),
  names(dat_policy)
)
stopifnot(length(covars) > 0)
X <- dat_policy[, covars, drop = FALSE]
X$logExposure <- log(pmax(dat_policy$Exposure, 1e-6))  # add as numeric feature

# 4) Fit your model (use slightly smaller epochs first to sanity-check)
res <- fit_zip_gamma_multi(
  X        = X,
  freq     = dat_policy$freq,
  sev_list = dat_policy$sev_list,
  kernel   = "exponential",
  delta    = 0.5,
  gammaL   = 0.5,
  hyper = list(
    freq_units       = c(128, 32),
    freq_activations = c("relu","relu"),
    freq_dropout     = c(0.15, 0.10),
    sev_units        = c(64, 32, 16, 8),
    sev_activations  = c("relu","relu","relu","relu"),
    sev_dropout      = c(0.15, 0.10, 0.05, 0.05),
    epochs           = 200,      # try 60–120 first; increase once it runs cleanly
    batch_size       = 1024,
    lr_freq          = 6e-3,
    lr_sev           = 1e-3
  ),
  max_iter = 8,    # a few IFM steps to start
  tol      = 1e-2,
  verbose  = 1
)

# 5) Quick checks
res$pi_hat
res$alpha_hat
res$omega_nn
head(res$preds$lambda)
head(res$preds$beta)




# ---- deps
library(ggplot2)
library(dplyr)
library(scales)
library(reticulate)

# Build a model.matrix for new data that matches training columns & scaling
.make_mm_scaled <- function(res, X_train, new_df) {
  # keep only columns used in training
  cols <- intersect(names(X_train), names(new_df))
  new_df <- new_df[, cols, drop = FALSE]
  X_train <- X_train[, cols, drop = FALSE]
  
  # Clean like fit: numerics -> median impute; chars -> factor with "__MISSING__"
  for (nm in names(new_df)) {
    if (is.numeric(X_train[[nm]])) {
      new_df[[nm]] <- suppressWarnings(as.numeric(new_df[[nm]]))
      med <- suppressWarnings(stats::median(X_train[[nm]], na.rm = TRUE))
      if (!is.finite(med)) med <- 0
      new_df[[nm]][!is.finite(new_df[[nm]]) | is.na(new_df[[nm]])] <- med
    } else {
      # template levels from training + "__MISSING__"
      tr_lev <- levels(factor(as.character(X_train[[nm]])))
      tr_lev <- unique(c(tr_lev, "__MISSING__"))
      v <- as.character(new_df[[nm]])
      v[is.na(v) | v == ""] <- "__MISSING__"
      v[!(v %in% tr_lev)] <- "__MISSING__"
      new_df[[nm]] <- factor(v, levels = tr_lev)
    }
  }
  
  # model.matrix with no intercept (same as fit)
  mm <- model.matrix(~ . - 1, data = new_df, na.action = na.pass)
  mm[!is.finite(mm)] <- 0
  
  # align columns to training feature space
  train_cols <- res$scaler_train$feature_names
  mm_full <- matrix(0, nrow = nrow(mm), ncol = length(train_cols),
                    dimnames = list(NULL, train_cols))
  common <- intersect(colnames(mm), train_cols)
  if (length(common) > 0) mm_full[, common] <- mm[, common, drop = FALSE]
  
  # scale with train scalers
  m  <- res$scaler_train$feat_mean
  sd <- res$scaler_train$feat_sd
  sd[!is.finite(sd) | sd == 0] <- 1
  mm_sc <- sweep(mm_full, 2, m, "-")
  mm_sc <- sweep(mm_sc, 2, sd, "/")
  mm_sc
}

# Predict E[N|X] for a new data frame (using the fitted frequency net)
.predict_EN <- function(res, mm_sc) {
  np <- reticulate::import("numpy")
  x  <- np$array(mm_sc, dtype = "float32")
  pred_sc <- as.vector(res$zip_model$predict(x, verbose = 0))
  # invert standardization from training
  lam  <- pred_sc * res$scaler_train$freq_sd + res$scaler_train$freq_mean
  EN   <- pmax((1 - res$pi_hat) * lam, 0)  # clamp negatives to 0
  EN
}

# ---- Main plotting helper
plot_EN_vs_X <- function(res, X, var, grid_n = 100) {
  stopifnot(var %in% names(X))
  # Baseline row: numeric -> median; factor/char -> most common
  base <- lapply(X, function(col) {
    if (is.numeric(col)) {
      med <- suppressWarnings(stats::median(col, na.rm = TRUE))
      if (!is.finite(med)) med <- 0
      med
    } else {
      v <- as.character(col)
      v[is.na(v) | v == ""] <- "__MISSING__"
      tab <- sort(table(v), decreasing = TRUE)
      names(tab)[1]
    }
  })
  base <- as.data.frame(base, stringsAsFactors = FALSE)
  
  # Grid for the focal variable
  if (is.numeric(X[[var]])) {
    rng <- stats::quantile(X[[var]], c(0.02, 0.98), na.rm = TRUE)
    grid_vals <- as.numeric(seq(rng[1], rng[2], length.out = grid_n))
    new_df <- base[rep(1, length(grid_vals)), , drop = FALSE]
    new_df[[var]] <- grid_vals
    mm_sc <- .make_mm_scaled(res, X, new_df)
    EN <- .predict_EN(res, mm_sc)
    dfp <- data.frame(x = grid_vals, EN = EN)
    
    ggplot(dfp, aes(x = x, y = EN)) +
      geom_line() +
      labs(x = var, y = expression(E[N ~ "|" ~ X]),
           title = bquote("Marginal " * E[N~"|"~.(var)] ~ "vs" ~ .(var))) +
      scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
      theme_minimal()
  } else {
    # treat as categorical
    v <- as.character(X[[var]])
    v[is.na(v) | v == ""] <- "__MISSING__"
    lev <- names(sort(table(v), decreasing = TRUE))
    new_df <- base[rep(1, length(lev)), , drop = FALSE]
    new_df[[var]] <- lev
    mm_sc <- .make_mm_scaled(res, X, new_df)
    EN <- .predict_EN(res, mm_sc)
    dfp <- data.frame(level = lev, EN = EN)
    
    ggplot(dfp, aes(x = reorder(level, EN), y = EN)) +
      geom_col(width = 0.7) +
      coord_flip() +
      labs(x = var, y = expression(E[N ~ "|" ~ X]),
           title = bquote("Marginal " * E[N~"|"~.(var)] ~ "by" ~ .(var))) +
      scale_y_continuous(labels = label_number(scale_cut = cut_short_scale())) +
      theme_minimal()
  }
}

# ---- Convenience: draw for every covariate in X
plot_EN_all <- function(res, X, grid_n = 100) {
  plots <- lapply(names(X), function(v) {
    try(plot_EN_vs_X(res, X, v, grid_n = grid_n), silent = TRUE)
  })
  # drop errors
  plots <- Filter(function(p) inherits(p, "ggplot"), plots)
  plots
}


library(patchwork)
library(ggplot2)

# plist from plot_EN_all(res, X, grid_n = 100)
k <- length(plist)

# chunk into groups of 2
chunks <- split(plist, ceiling(seq_along(plist) / 2))

# build each page (2-wide), add a clean title
pages <- lapply(seq_along(chunks), function(i){
  wrap_plots(chunks[[i]], ncol = 2, guides = "collect") +
    plot_annotation(
      title = paste0("Marginal E[N|X] by covariate — page ", i, " of ", length(chunks)),
      theme = theme_minimal(base_size = 13) +
        theme(plot.title = element_text(face = "bold"))
    )
})

# Preview page 1
print(pages[[1]])

# Save as a multi-page PDF
pdf("EN_marginals_paginated.pdf", width = 12, height = 6)
for (pg in pages) print(pg)
dev.off()

# (Optional) also save each page as a PNG
for (i in seq_along(pages)) {
  ggsave(sprintf("EN_marginals_page_%02d.png", i), pages[[i]], width = 12, bg = "white", height = 6, dpi = 300)
}


# ==== Predictive aggregate loss on the validation set via bootstrap ====
# Inputs:
#   res      : object from fit_zip_gamma_multi(...)
#   X,freq,sev_list : the same objects you used to fit (for actual loss)
#   B        : number of bootstrap replicates
#   seed     : RNG seed
# Returns: list(sim = vector, actual = number, stats = data.frame)
predict_val_aggregate_zip_gamma <- function(res, X, freq, sev_list,
                                            B = 1000, seed = 369,
                                            verbose_every = 100) {
  stopifnot(!is.null(res$indices$val_id))
  val_id <- res$indices$val_id
  m <- length(val_id)
  if (m == 0) stop("Validation set is empty in res$indices$val_id.")
  
  # --- pull model predictions for the val set ---
  # res$preds$lambda is the *unconditional* ZIP mean E[N|X] = (1 - pi)*lambda_pois
  lam_uncond <- as.numeric(res$preds$lambda[val_id])
  lam_uncond[!is.finite(lam_uncond) | lam_uncond < 0] <- 0
  
  denom <- max(1 - res$pi_hat, 1e-8)
  lam_pois <- lam_uncond / denom                      # Poisson mean in the nonzero state
  lam_pois[!is.finite(lam_pois) | lam_pois < 0] <- 0
  
  beta_scale <- as.numeric(res$preds$beta[val_id])    # Gamma scale from the NN
  beta_scale[!is.finite(beta_scale) | beta_scale <= 0] <- 1e-8
  alpha <- ifelse(is.finite(res$alpha_hat) && res$alpha_hat > 0, res$alpha_hat, 1.0)
  
  # --- actual aggregate on validation set ---
  sev_val <- sev_list[val_id]
  actual_loss <- sum(unlist(sev_val), na.rm = TRUE)
  
  # --- bootstrap predictive distribution ---
  set.seed(seed)
  sim_total <- numeric(B)
  for (b in seq_len(B)) {
    # resample policies WITH replacement from validation set
    pick <- sample.int(m, m, replace = TRUE)
    
    # zero-inflation state: Z=1 means "active" (not structural zero)
    Z <- rbinom(m, size = 1, prob = denom)
    N <- Z * rpois(m, lambda = lam_pois[pick])
    
    # severities
    total_b <- 0.0
    if (any(N > 0)) {
      for (i in which(N > 0)) {
        n_i <- N[i]
        beta_i <- beta_scale[pick[i]]
        # sum of n_i i.i.d. Gamma(shape=alpha, scale=beta_i)
        total_b <- total_b + sum(stats::rgamma(n_i, shape = alpha, scale = beta_i))
      }
    }
    sim_total[b] <- total_b
    
    if (verbose_every > 0 && (b %% verbose_every == 0))
      cat("Bootstrap", b, "of", B, "\n")
  }
  
  # simple summary
  qs <- c(0.025, 0.5, 0.975)
  stats <- data.frame(
    mean = mean(sim_total),
    sd   = sd(sim_total),
    q2.5 = unname(quantile(sim_total, qs[1])),
    q50  = unname(quantile(sim_total, qs[2])),
    q97.5= unname(quantile(sim_total, qs[3]))
  )
  
  # --- plot ---
  hist(sim_total, breaks = 40,
       main = "Predictive Distribution vs Actual Aggregate Loss (NN-Sarmanov)",
       xlab = "Aggregate Loss",
       col = "lightblue", border = "white")
  abline(v = mean(sim_total), col = "red", lwd = 2)
  legend("topright", legend = "Actual Aggregate Loss", col = "red", lwd = 2, bty = "n")
  
  invisible(list(sim = sim_total, actual = actual_loss, stats = stats))
}

# res, X, freq, sev_list already exist from your fit
out <- predict_val_aggregate_zip_gamma(res, X, freq, sev_list, B = 3000, seed = 369)
out$stats
out$actual


