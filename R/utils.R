
# ---- Utilities (no library() calls; use ::) ----

.mm <- function(X) {
  DF <- as.data.frame(X)
  # include intercept then drop it
  mm <- stats::model.matrix(~ . , data = DF)
  if("(Intercept)" %in% colnames(mm)) {
    mm <- mm[, setdiff(colnames(mm), "(Intercept)"), drop = FALSE]
  }
  mm <- as.matrix(mm)
  storage.mode(mm) <- "double"
  mm
}

.scale_cols <- function(M) {
  M <- as.matrix(M)
  center <- colMeans(M)
  scalev <- apply(M, 2, stats::sd)
  scalev[!is.finite(scalev) | scalev == 0] <- 1
  list(
    X = scale(M, center = center, scale = scalev),
    center = center,
    scale = scalev
  )
}

.clip_pos <- function(x, eps = 1e-8) {
  x[!is.finite(x)] <- eps
  x <- ifelse(x < eps, eps, x)
  as.numeric(x)
}

.alpha_from_moments <- function(sev) {
  m <- mean(sev[sev>0])
  v <- stats::var(sev[sev>0])
  if (!is.finite(m) || !is.finite(v) || v <= 0) return(1.0)
  max((m^2) / v, 0.5)
}

.fit_alpha_gamma <- function(sev) {
  sev <- sev[is.finite(sev) & sev > 0]
  if (length(sev) < 10) return(.alpha_from_moments(sev))
  out <- try(MASS::fitdistr(sev, densfun = "gamma"), silent = TRUE)
  if (inherits(out, "try-error")) return(.alpha_from_moments(sev))
  a <- out$estimate[["shape"]]
  if (!is.finite(a) || a <= 0) a <- .alpha_from_moments(sev)
  a
}

.safe_omega <- function(z) {
  z <- z[is.finite(z) & !is.na(z)]
  if (length(z) == 0 || all(z == 0)) return(0)
  L <- if (any(z > 0)) max(-1/z[z > 0]) else -Inf
  U <- if (any(z < 0)) min(-1/z[z < 0]) else  Inf
  L <- if (is.finite(L)) L + .Machine$double.eps else -1 + 1e-6
  U <- if (is.finite(U)) U - .Machine$double.eps else  1 - 1e-6
  if (!(L < U)) return(0)
  negll <- function(w) {
    W <- 1 + w * z
    if (any(!is.finite(W) | W <= 0)) return(Inf)
    -sum(log(W))
  }
  stats::optimize(negll, interval = c(L, U))$minimum
}

.require_keras <- function() {
  if (!requireNamespace("keras", quietly = TRUE) ||
      !requireNamespace("tensorflow", quietly = TRUE)) {
    stop("keras/tensorflow are not available. Install with: install.packages(c('keras','tensorflow')); keras::install_keras()", call. = FALSE)
  }
  invisible(TRUE)
}

.build_mlp <- function(input_dim, units, activations, dropout, lr = 1e-3, loss = "mse") {
  .require_keras()
  keras::k_clear_session()
  stopifnot(length(units) == length(activations))
  m <- keras::keras_model_sequential()
  # first layer with input_shape
  m %>% keras::layer_dense(units = units[1], activation = activations[1], input_shape = input_dim)
  if (length(dropout) >= 1 && dropout[1] > 0) m %>% keras::layer_dropout(rate = dropout[1])
  if (length(units) > 1) {
    for (i in 2:length(units)) {
      m %>% keras::layer_dense(units = units[i], activation = activations[i])
      if (length(dropout) >= i && dropout[i] > 0) m %>% keras::layer_dropout(rate = dropout[i])
    }
  }
  # output layer linear
  m %>% keras::layer_dense(units = 1, activation = "linear")
  m %>% keras::compile(optimizer = keras::optimizer_adam(learning_rate = lr), loss = loss)
  m
}

# Kernels
.get_phi_gamma <- function(x, alpha, beta, kernel = c("standardized","exponential"), gammaL = 0.5) {
  kernel <- match.arg(kernel)
  if (kernel == "exponential") {
    (exp(-gammaL * x) - (1 + beta * gammaL)^(-alpha))
  } else {
    mu <- alpha * beta; sx <- sqrt(alpha) * beta
    (x - mu) / sx
  }
}

.get_phi_logn <- function(logx, mu, sigma, kernel = c("standardized","exponential"), gammaL = 0.5) {
  kernel <- match.arg(kernel)
  if (kernel == "exponential") {
    exp(-gammaL * exp(logx)) - exp(mu * (-gammaL) + 0.5 * sigma^2 * gammaL^2)
  } else {
    (logx - mu) / sigma
  }
}

.get_psi <- function(n, m, kernel = c("standardized","exponential"), delta = 0.5) {
  kernel <- match.arg(kernel)
  m <- .clip_pos(m)
  if (kernel == "exponential") {
    exp(-delta * n) - exp(-delta * m)
  } else {
    (n - m) / sqrt(m)
  }
}

.split_train_val <- function(N, val_frac = 0.1, seed = 369) {
  set.seed(seed)
  ix <- sample.int(N)
  n_val <- max(1L, as.integer(N * val_frac))
  list(val = ix[seq_len(n_val)], train = ix[(n_val+1L):N])
}
