
# ---- GLM Sarmanov fitters (multi-covariate) ----

# Poisson-Gamma (GLM)
fit_poisson_gamma_glm_sarmanov <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  max_iter = 50, tol = 1e-3, verbose = 1
){
  kernel <- match.arg(kernel)
  mm <- .mm(X)
  DF <- data.frame(count = as.numeric(freq), mm)
  form_f <- stats::as.formula(paste("count ~", paste(colnames(mm), collapse = " + ")))
  freq_glm <- stats::glm(form_f, family = stats::poisson(link = "log"), data = DF)
  lambda_hat <- as.numeric(stats::predict(freq_glm, type = "response"))

  # per-policy means for positive freq
  policy_idx <- which(freq > 0)
  sev_mean <- vapply(sev_list[policy_idx], function(v) mean(v[v>0]), numeric(1))
  sev_all  <- unlist(sev_list)
  alpha_hat <- .fit_alpha_gamma(sev_all)

  DFp <- data.frame(sev = sev_mean, mm[policy_idx,,drop=FALSE])
  form_s <- stats::as.formula(paste("sev ~", paste(colnames(mm), collapse = " + ")))
  sev_glm <- stats::glm(form_s, family = stats::Gamma(link = "log"), data = DFp)
  coefs <- stats::coef(sev_glm)

  omega <- 0
  for (iter in seq_len(max_iter)) {
    mu_hat <- as.numeric(exp(stats::model.matrix(form_s, DFp) %*% coefs))
    beta_hat <- mu_hat / alpha_hat
    psi <- .get_psi(freq[policy_idx], lambda_hat[policy_idx], kernel, delta)
    phi <- .get_phi_gamma(sev_mean, alpha_hat, beta_hat, kernel, gammaL)
    z <- psi * phi
    omega_new <- .safe_omega(z)

    # update severity coefs by maximizing weighted loglik + log(1+omega*z)
    negll <- function(b) {
      mu <- as.numeric(exp(stats::model.matrix(form_s, DFp) %*% b))
      beta <- mu / alpha_hat
      W <- 1 + omega_new * psi * .get_phi_gamma(sev_mean, alpha_hat, beta, kernel, gammaL)
      if (any(!is.finite(W) | W <= 0)) return(1e10)
      ll1 <- sum(stats::dgamma(sev_mean, shape = alpha_hat, scale = beta, log = TRUE))
      -(ll1 + sum(log(W)))
    }
    opt <- try(stats::optim(coefs, negll, method = "BFGS"), silent = TRUE)
    if (!inherits(opt, "try-error")) coefs <- opt$par

    # refit frequency glm (without omega term)
    freq_glm <- stats::glm(form_f, family = stats::poisson(link = "log"), data = DF)
    lambda_hat <- as.numeric(stats::predict(freq_glm, type = "response"))

    if (verbose > 0) message(sprintf("Iter %d: omega=%.5f", iter, omega_new))
    if (max(abs(omega_new - omega)) < tol) { omega <- omega_new; break }
    omega <- omega_new
  }

  list(freq_glm = freq_glm, sev_glm = coefs, alpha_hat = alpha_hat, omega = omega,
       terms_sev = names(coefs))
}

# ZIP-Gamma (GLM)
fit_zip_gamma_glm_sarmanov <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  max_iter = 50, tol = 1e-3, verbose = 1
){
  kernel <- match.arg(kernel)
  mm <- .mm(X)
  DF <- data.frame(count = as.integer(freq), mm)
  # intercept-only ZIP for pi
  zip_fit <- try(pscl::zeroinfl(count ~ 1 | 1, data = DF, dist = "poisson"), silent = TRUE)
  if (inherits(zip_fit, "try-error")) stop("pscl::zeroinfl failed; install pscl.", call. = FALSE)
  pi_hat <- plogis(stats::coef(zip_fit)[["zero_(Intercept)"]])

  # Poisson GLM for lambda ~ X
  form_f <- stats::as.formula(paste("count ~", paste(colnames(mm), collapse = " + ")))
  freq_glm <- stats::glm(form_f, family = stats::poisson, data = DF)
  lambda_hat <- as.numeric(stats::predict(freq_glm, type = "response"))
  lambda_adj <- (1 - pi_hat) * lambda_hat

  policy_idx <- which(freq > 0)
  sev_mean <- vapply(sev_list[policy_idx], function(v) mean(v[v>0]), numeric(1))
  sev_all  <- unlist(sev_list)
  alpha_hat <- .fit_alpha_gamma(sev_all)

  DFp <- data.frame(sev = sev_mean, mm[policy_idx,,drop=FALSE])
  form_s <- stats::as.formula(paste("sev ~", paste(colnames(mm), collapse = " + ")))
  sev_glm <- stats::glm(form_s, family = stats::Gamma(link = "log"), data = DFp)
  coefs <- stats::coef(sev_glm)

  omega <- 0
  for (iter in seq_len(max_iter)) {
    mu_hat <- as.numeric(exp(stats::model.matrix(form_s, DFp) %*% coefs))
    beta_hat <- mu_hat / alpha_hat
    psi <- .get_psi(freq[policy_idx], lambda_adj[policy_idx], kernel, delta)
    phi <- .get_phi_gamma(sev_mean, alpha_hat, beta_hat, kernel, gammaL)
    z <- psi * phi
    omega_new <- .safe_omega(z)

    negll <- function(b) {
      mu <- as.numeric(exp(stats::model.matrix(form_s, DFp) %*% b))
      beta <- mu / alpha_hat
      W <- 1 + omega_new * psi * .get_phi_gamma(sev_mean, alpha_hat, beta, kernel, gammaL)
      if (any(!is.finite(W) | W <= 0)) return(1e10)
      ll1 <- sum(stats::dgamma(sev_mean, shape = alpha_hat, scale = beta, log = TRUE))
      -(ll1 + sum(log(W)))
    }
    opt <- try(stats::optim(coefs, negll, method = "BFGS"), silent = TRUE)
    if (!inherits(opt, "try-error")) coefs <- opt$par

    freq_glm <- stats::glm(form_f, family = stats::poisson, data = DF)
    lambda_hat <- as.numeric(stats::predict(freq_glm, type = "response"))
    lambda_adj <- (1 - pi_hat) * lambda_hat

    if (verbose > 0) message(sprintf("Iter %d: omega=%.5f", iter, omega_new))
    if (max(abs(omega_new - omega)) < tol) { omega <- omega_new; break }
    omega <- omega_new
  }

  list(freq_glm = freq_glm, sev_glm = coefs, alpha_hat = alpha_hat, omega = omega, pi_hat = pi_hat,
       terms_sev = names(coefs))
}

# NB-Gamma (GLM)
fit_nb_gamma_glm_sarmanov <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  max_iter = 50, tol = 1e-3, verbose = 1
){
  kernel <- match.arg(kernel)
  mm <- .mm(X)
  DF <- data.frame(count = as.numeric(freq), mm)
  form_f <- stats::as.formula(paste("count ~", paste(colnames(mm), collapse = " + ")))
  nb_fit <- try(MASS::glm.nb(form_f, data = DF), silent = TRUE)
  if (inherits(nb_fit, "try-error")) stop("MASS::glm.nb failed; install MASS.", call. = FALSE)
  mu_hat <- as.numeric(stats::predict(nb_fit, type = "response"))

  policy_idx <- which(freq > 0)
  sev_mean <- vapply(sev_list[policy_idx], function(v) mean(v[v>0]), numeric(1))
  sev_all  <- unlist(sev_list)
  alpha_hat <- .fit_alpha_gamma(sev_all)

  DFp <- data.frame(sev = sev_mean, mm[policy_idx,,drop=FALSE])
  form_s <- stats::as.formula(paste("sev ~", paste(colnames(mm), collapse = " + ")))
  sev_glm <- stats::glm(form_s, family = stats::Gamma(link = "log"), data = DFp)
  coefs <- stats::coef(sev_glm)

  omega <- 0
  for (iter in seq_len(max_iter)) {
    mu_sev <- as.numeric(exp(stats::model.matrix(form_s, DFp) %*% coefs))
    beta_hat <- mu_sev / alpha_hat
    psi <- .get_psi(freq[policy_idx], mu_hat[policy_idx], kernel, delta)
    phi <- .get_phi_gamma(sev_mean, alpha_hat, beta_hat, kernel, gammaL)
    z <- psi * phi
    omega_new <- .safe_omega(z)

    negll <- function(b) {
      mu <- as.numeric(exp(stats::model.matrix(form_s, DFp) %*% b))
      beta <- mu / alpha_hat
      W <- 1 + omega_new * psi * .get_phi_gamma(sev_mean, alpha_hat, beta, kernel, gammaL)
      if (any(!is.finite(W) | W <= 0)) return(1e10)
      ll1 <- sum(stats::dgamma(sev_mean, shape = alpha_hat, scale = beta, log = TRUE))
      -(ll1 + sum(log(W)))
    }
    opt <- try(stats::optim(coefs, negll, method = "BFGS"), silent = TRUE)
    if (!inherits(opt, "try-error")) coefs <- opt$par

    nb_fit <- MASS::glm.nb(form_f, data = DF)
    mu_hat <- as.numeric(stats::predict(nb_fit, type = "response"))

    if (verbose > 0) message(sprintf("Iter %d: omega=%.5f", iter, omega_new))
    if (max(abs(omega_new - omega)) < tol) { omega <- omega_new; break }
    omega <- omega_new
  }

  list(freq_glm = nb_fit, sev_glm = coefs, alpha_hat = alpha_hat, omega = omega, terms_sev = names(coefs))
}

# Poisson-Lognormal (GLM)
fit_poisson_lognormal_glm_sarmanov <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  max_iter = 50, tol = 1e-2, verbose = 1
){
  kernel <- match.arg(kernel)
  mm <- .mm(X)
  DF <- data.frame(count = as.numeric(freq), mm)
  form_f <- stats::as.formula(paste("count ~", paste(colnames(mm), collapse = " + ")))
  freq_glm <- stats::glm(form_f, family = stats::poisson, data = DF)
  lambda_hat <- as.numeric(stats::predict(freq_glm, type = "response"))

  policy_idx <- which(freq > 0)
  sev_mean <- vapply(sev_list[policy_idx], function(v) mean(v[v>0]), numeric(1))
  sev_all  <- unlist(sev_list)
  sigma_hat <- stats::sd(log(sev_all[sev_all>0]))

  DFp <- data.frame(logsev = log(sev_mean), mm[policy_idx,,drop=FALSE])
  form_s <- stats::as.formula(paste("logsev ~", paste(colnames(mm), collapse = " + ")))
  sev_lm <- stats::lm(form_s, data = DFp)
  b <- stats::coef(sev_lm)

  omega <- 0
  for (iter in seq_len(max_iter)) {
    mu_log <- as.numeric(stats::model.matrix(form_s, DFp) %*% b)
    psi <- .get_psi(freq[policy_idx], lambda_hat[policy_idx], kernel, delta)
    phi <- .get_phi_logn(log(sev_mean), mu_log, sigma_hat, kernel, gammaL)
    z <- psi * phi
    omega_new <- .safe_omega(z)

    negll <- function(bb) {
      mu <- as.numeric(stats::model.matrix(form_s, DFp) %*% bb)
      W <- 1 + omega_new * psi * .get_phi_logn(log(sev_mean), mu, sigma_hat, kernel, gammaL)
      if (any(!is.finite(W) | W <= 0)) return(1e10)
      ll1 <- sum(stats::dnorm(log(sev_mean), mean = mu, sd = sigma_hat, log = TRUE))
      -(ll1 + sum(log(W)))
    }
    opt <- try(stats::optim(b, negll, method = "BFGS"), silent = TRUE)
    if (!inherits(opt, "try-error")) b <- opt$par

    freq_glm <- stats::glm(form_f, family = stats::poisson, data = DF)
    lambda_hat <- as.numeric(stats::predict(freq_glm, type = "response"))

    if (verbose > 0) message(sprintf("Iter %d: omega=%.5f", iter, omega_new))
    if (max(abs(omega_new - omega)) < tol) { omega <- omega_new; break }
    omega <- omega_new
  }

  list(freq_glm = freq_glm, sev_glm = b, sigma_hat = sigma_hat, omega = omega, terms_sev = names(b))
}

# ZIP-Lognormal (GLM)
fit_zip_lognormal_glm_sarmanov <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  max_iter = 50, tol = 1e-2, verbose = 1
){
  kernel <- match.arg(kernel)
  mm <- .mm(X)
  DF <- data.frame(count = as.integer(freq), mm)

  zip_fit <- try(pscl::zeroinfl(count ~ 1 | 1, data = DF, dist = "poisson"), silent = TRUE)
  if (inherits(zip_fit, "try-error")) stop("pscl::zeroinfl failed; install pscl.", call. = FALSE)
  pi_hat <- plogis(stats::coef(zip_fit)[["zero_(Intercept)"]])

  form_f <- stats::as.formula(paste("count ~", paste(colnames(mm), collapse = " + ")))
  freq_glm <- stats::glm(form_f, family = stats::poisson, data = DF)
  lambda_hat <- as.numeric(stats::predict(freq_glm, type = "response"))

  policy_idx <- which(freq > 0)
  sev_mean <- vapply(sev_list[policy_idx], function(v) mean(v[v>0]), numeric(1))
  sev_all  <- unlist(sev_list)
  sigma_hat <- stats::sd(log(sev_all[sev_all>0]))

  DFp <- data.frame(logsev = log(sev_mean), mm[policy_idx,,drop=FALSE])
  form_s <- stats::as.formula(paste("logsev ~", paste(colnames(mm), collapse = " + ")))
  sev_lm <- stats::lm(form_s, data = DFp)
  b <- stats::coef(sev_lm)

  omega <- 0
  for (iter in seq_len(max_iter)) {
    mu_log <- as.numeric(stats::model.matrix(form_s, DFp) %*% b)
    psi <- .get_psi(freq[policy_idx], (1 - pi_hat) * lambda_hat[policy_idx], kernel, delta)
    phi <- .get_phi_logn(log(sev_mean), mu_log, sigma_hat, kernel, gammaL)
    z <- psi * phi
    omega_new <- .safe_omega(z)

    negll <- function(bb) {
      mu <- as.numeric(stats::model.matrix(form_s, DFp) %*% bb)
      W <- 1 + omega_new * psi * .get_phi_logn(log(sev_mean), mu, sigma_hat, kernel, gammaL)
      if (any(!is.finite(W) | W <= 0)) return(1e10)
      ll1 <- sum(stats::dnorm(log(sev_mean), mean = mu, sd = sigma_hat, log = TRUE))
      -(ll1 + sum(log(W)))
    }
    opt <- try(stats::optim(b, negll, method = "BFGS"), silent = TRUE)
    if (!inherits(opt, "try-error")) b <- opt$par

    freq_glm <- stats::glm(form_f, family = stats::poisson, data = DF)
    lambda_hat <- as.numeric(stats::predict(freq_glm, type = "response"))

    if (verbose > 0) message(sprintf("Iter %d: omega=%.5f", iter, omega_new))
    if (max(abs(omega_new - omega)) < tol) { omega <- omega_new; break }
    omega <- omega_new
  }

  list(freq_glm = freq_glm, sev_glm = b, sigma_hat = sigma_hat, omega = omega, pi_hat = pi_hat, terms_sev = names(b))
}

# NB-Lognormal (GLM)
fit_nb_lognormal_glm_sarmanov <- function(
  X, freq, sev_list,
  kernel = c("standardized","exponential"),
  delta = 0.5, gammaL = 0.5,
  max_iter = 50, tol = 1e-2, verbose = 1
){
  kernel <- match.arg(kernel)
  mm <- .mm(X)
  DF <- data.frame(count = as.numeric(freq), mm)
  form_f <- stats::as.formula(paste("count ~", paste(colnames(mm), collapse = " + ")))
  nb_fit <- try(MASS::glm.nb(form_f, data = DF), silent = TRUE)
  if (inherits(nb_fit, "try-error")) stop("MASS::glm.nb failed; install MASS.", call. = FALSE)
  mu_nb <- as.numeric(stats::predict(nb_fit, type = "response"))

  policy_idx <- which(freq > 0)
  sev_mean <- vapply(sev_list[policy_idx], function(v) mean(v[v>0]), numeric(1))
  sev_all  <- unlist(sev_list)
  sigma_hat <- stats::sd(log(sev_all[sev_all>0]))

  DFp <- data.frame(logsev = log(sev_mean), mm[policy_idx,,drop=FALSE])
  form_s <- stats::as.formula(paste("logsev ~", paste(colnames(mm), collapse = " + ")))
  sev_lm <- stats::lm(form_s, data = DFp)
  b <- stats::coef(sev_lm)

  omega <- 0
  for (iter in seq_len(max_iter)) {
    mu_log <- as.numeric(stats::model.matrix(form_s, DFp) %*% b)
    psi <- .get_psi(n = freq[policy_idx], m = mu_nb[policy_idx], kernel = kernel, delta = delta)
    phi <- .get_phi_logn(log(sev_mean), mu_log, sigma_hat, kernel, gammaL)
    z <- psi * phi
    omega_new <- .safe_omega(z)

    negll <- function(bb) {
      mu <- as.numeric(stats::model.matrix(form_s, DFp) %*% bb)
      W <- 1 + omega_new * psi * .get_phi_logn(log(sev_mean), mu, sigma_hat, kernel, gammaL)
      if (any(!is.finite(W) | W <= 0)) return(1e10)
      ll1 <- sum(stats::dnorm(log(sev_mean), mean = mu, sd = sigma_hat, log = TRUE))
      -(ll1 + sum(log(W)))
    }
    opt <- try(stats::optim(b, negll, method = "BFGS"), silent = TRUE)
    if (!inherits(opt, "try-error")) b <- opt$par

    nb_fit <- MASS::glm.nb(form_f, data = DF)
    mu_nb <- as.numeric(stats::predict(nb_fit, type = "response"))

    if (verbose > 0) message(sprintf("Iter %d: omega=%.5f", iter, omega_new))
    if (max(abs(omega_new - omega)) < tol) { omega <- omega_new; break }
    omega <- omega_new
  }

  list(freq_glm = nb_fit, sev_glm = b, sigma_hat = sigma_hat, omega = omega, terms_sev = names(b))
}
