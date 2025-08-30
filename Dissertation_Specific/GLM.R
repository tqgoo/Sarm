fit_poisson_gamma_glm_sarmanov <- function(
    X, freq, sev_list,
    kernel = "exponential",      # or "exponential"
    delta = 0.5,                  # for exponential psi
    gammaL = 0.5,                 # for exponential phi
    max_iter = 50,
    tol = 1e-3,
    verbose = 2
) {
  require(MASS)
  age <- as.numeric(X[,1])
  poisson_df <- data.frame(age = age, count = freq)
  
  # --- Per-policy aggregation for severity
  policy_idx <- which(freq > 0)
  age_pol    <- age[policy_idx]
  n_pol      <- freq[policy_idx]
  sev_mean   <- sapply(sev_list[policy_idx], mean)
  sev        <- unlist(sev_list)
  
  # Gamma shape parameter (fixed for IFM)
  alpha_hat <- fitdistr(sev, "gamma")$estimate["shape"]
  
  # --- Kernels (standardized or exponential) ---
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
  
  # --- Initial regression fits
  freq_glm <- glm(count ~ age, family=poisson, data=poisson_df)
  lambda_hat <- predict(freq_glm, type="response")
  sev_df  <- data.frame(age=age_pol, sev=sev_mean)
  sev_glm <- glm(sev ~ age, family=Gamma(link="log"), data=sev_df)
  gamma   <- coef(sev_glm)
  
  omega <- 0.1
  for(iter in 1:max_iter) {
    omega_old <- omega
    gamma_old <- gamma
    
    # psi and phi for claims-level (for log-lik in dependence)
    psi_vals <- get_psi_vector(freq, lambda_hat, kernel, delta)
    psi_rep  <- psi_vals[policy_idx]
    mu_hat_claims <- exp(gamma[1] + gamma[2]*age_pol)
    b_claims      <- mu_hat_claims / alpha_hat
    phi_vals      <- get_phi_vector(sev_mean, alpha_hat, b_claims, kernel, gammaL)
    z <- psi_rep * phi_vals
    
    L <- if(any(z>0)) max(-1/z[z>0]) else -Inf
    U <- if(any(z<0)) min(-1/z[z<0]) else Inf
    L <- L + .Machine$double.eps;  U <- U - .Machine$double.eps
    negll_omega <- function(w) { -sum(log(1 + w * z)) }
    omega <- optimize(negll_omega, interval=c(L,U))$minimum
    
    # Update severity GLM (policy level)
    negll_gamma <- function(g) {
      mu_hat    <- exp(g[1] + g[2]*age_pol)
      b_hat     <- mu_hat / alpha_hat
      phi_ij <- get_phi_vector(sev_mean, alpha_hat, b_hat, kernel, gammaL)
      W      <- 1 + omega * psi_rep * phi_ij
      if(!all(is.finite(W)) || any(W <= 0)) return(1e10)
      ll1 <- sum(dgamma(sev_mean, shape=alpha_hat, scale=b_hat, log=TRUE))
      ll2 <- sum(log(W))
      -(ll1 + ll2)
    }
    optg  <- optim(gamma, negll_gamma, method="BFGS")
    gamma <- optg$par
    
    # Refit frequency GLM
    freq_glm <- glm(count ~ age, family=poisson, data=poisson_df)
    lambda_hat <- predict(freq_glm, type="response")
    
    if(max(abs(omega-omega_old), abs(gamma-gamma_old)) < tol) {
      if(verbose > 0) cat("Converged! Breaking.\n")
      break
    }
    if(verbose > 0) cat("GLM-Sarmanov omega:", round(omega,4), "\n")
  }
  
  # Return everything you need for downstream simulation, plotting, or correlation calculation:
  list(
    freq_glm = freq_glm,
    sev_glm = gamma,
    alpha_hat = alpha_hat,
    omega = omega
  )
}






fit_zip_gamma_glm_sarmanov <- function(
    X, freq, sev_list,
    kernel = "standardized",      # or "exponential"
    delta = 0.5,                  # for exponential psi, if needed
    gammaL = 0.5,                 # for exponential phi
    max_iter = 50,
    tol = 1e-3,
    verbose = 2
) {
  require(pscl)
  require(MASS)
  
  age <- as.numeric(X[,1])
  zip_df <- data.frame(age = age, count = as.integer(freq))
  
  # 1. Fit initial ZIP (intercept only)
  fit_zip <- pscl::zeroinfl(count ~ 1 | 1, data = zip_df, dist = "poisson")
  pi_hat <- plogis(coef(fit_zip)["zero_(Intercept)"])
  lambda_zip <- zip_df$count / (1 - pi_hat)
  lambda_zip[is.na(lambda_zip) | is.infinite(lambda_zip)] <- 0
  
  # Per-policy aggregation for severity
  policy_idx <- which(freq > 0)
  age_pol    <- age[policy_idx]
  n_pol      <- freq[policy_idx]
  sev_mean   <- sapply(sev_list[policy_idx], mean)
  sev        <- unlist(sev_list)
  
  # Gamma shape parameter (global)
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
  
  # Initial regression fits
  freq_glm <- glm(count ~ age, family=poisson, data=zip_df)
  lambda_hat <- predict(freq_glm, type="response")
  lambda_hat_adj <- (1 - pi_hat) * lambda_hat
  
  sev_df  <- data.frame(age=age_pol, sev=sev_mean)
  sev_glm <- glm(sev ~ age, family=Gamma(link="log"), data=sev_df)
  gamma   <- coef(sev_glm)
  
  omega <- 0.1
  for(iter in 1:max_iter) {
    omega_old <- omega
    gamma_old <- gamma
    
    # psi and phi for claims-level (for log-lik in dependence)
    psi_vals <- get_psi_vector(freq, lambda_hat_adj, kernel, delta)
    psi_rep  <- psi_vals[policy_idx]
    mu_hat_claims <- exp(gamma[1] + gamma[2]*age_pol)
    b_claims      <- mu_hat_claims / alpha_hat
    phi_vals      <- get_phi_vector(sev_mean, alpha_hat, b_claims, kernel, gammaL)
    z <- psi_rep * phi_vals
    
    L <- if(any(z>0)) max(-1/z[z>0]) else -Inf
    U <- if(any(z<0)) min(-1/z[z<0]) else Inf
    L <- L + .Machine$double.eps;  U <- U - .Machine$double.eps
    negll_omega <- function(w) { -sum(log(1 + w * z)) }
    omega <- optimize(negll_omega, interval=c(L,U))$minimum
    
    # Update severity GLM (policy level)
    negll_gamma <- function(g) {
      mu_hat    <- exp(g[1] + g[2]*age_pol)
      b_hat     <- mu_hat / alpha_hat
      phi_ij <- get_phi_vector(sev_mean, alpha_hat, b_hat, kernel, gammaL)
      W      <- 1 + omega * psi_rep * phi_ij
      if(!all(is.finite(W)) || any(W <= 0)) return(1e10)
      ll1 <- sum(dgamma(sev_mean, shape=alpha_hat, scale=b_hat, log=TRUE))
      ll2 <- sum(log(W))
      -(ll1 + ll2)
    }
    optg  <- optim(gamma, negll_gamma, method="BFGS", control=list(reltol=tol, maxit=1000))
    gamma <- optg$par
    
    # Refit frequency GLM (Poisson on all, still no Sarmanov term here)
    freq_glm <- glm(count ~ age, family=poisson, data=zip_df)
    lambda_hat <- predict(freq_glm, type="response")
    lambda_hat_adj <- (1 - pi_hat) * lambda_hat
    
    if(max(abs(omega-omega_old), abs(gamma-gamma_old)) < tol) {
      if(verbose > 0) cat("Converged! Breaking.\n")
      break
    }
    if(verbose > 0) cat("GLM-Sarmanov omega:", round(omega,4), "\n")
  }
  
  list(
    freq_glm = freq_glm,
    sev_glm = gamma,
    alpha_hat = alpha_hat,
    omega = omega,
    pi_hat = pi_hat
  )
}


fit_nb_gamma_glm_sarmanov <- function(
    X, freq, sev_list,
    kernel = "standardized",      # or "exponential"
    delta = 0.5,                  # for exponential psi, if needed
    gammaL = 0.5,                 # for exponential phi
    max_iter = 50,
    tol = 1e-3,
    verbose = 2
) {
  require(MASS)
  age <- as.numeric(X[,1])
  nb_df <- data.frame(age = age, count = freq)
  
  # --- Per-policy aggregation for severity ---
  policy_idx <- which(freq > 0)
  age_pol    <- age[policy_idx]
  n_pol      <- freq[policy_idx]
  sev_mean   <- sapply(sev_list[policy_idx], mean)
  sev        <- unlist(sev_list)
  
  # Gamma shape parameter (fixed for IFM)
  alpha_hat <- fitdistr(sev, "gamma")$estimate["shape"]
  
  # --- Kernel helpers ---
  phi_exp_gamma <- function(x, alpha, beta, gamma) exp(-gamma * x) - (1 + beta * gamma)^(-alpha)
  phi_std_gamma <- function(x, mu, sx) (x - mu) / sx
  get_phi_vector <- function(x, alpha, beta, kernel, gammaL) {
    if(kernel == "exponential") {
      phi_exp_gamma(x, alpha, beta, gammaL)
    } else {
      phi_std_gamma(x, alpha * beta, sqrt(alpha) * beta)
    }
  }
  get_psi_vector <- function(n, mu_nb, kernel, delta) {
    if(kernel == "exponential") {
      exp(-delta * n) - exp(-delta * mu_nb)
    } else {
      (n - mu_nb) / sqrt(mu_nb)
    }
  }
  
  # --- Initial NB regression fit
  freq_glm <- glm.nb(count ~ age, data=nb_df)
  r_hat <- freq_glm$theta
  mu_hat <- predict(freq_glm, type="response")
  p_hat <- r_hat / (r_hat + mu_hat)
  
  # --- Initial severity GLM (policy level) ---
  sev_df <- data.frame(age=age_pol, sev=sev_mean)
  sev_glm <- glm(sev ~ age, family=Gamma(link="log"), data=sev_df)
  gamma   <- coef(sev_glm)
  
  omega <- 0.1
  for(iter in 1:max_iter) {
    omega_old <- omega
    gamma_old <- gamma
    
    # --- psi and phi for claims-level ---
    psi_vals <- get_psi_vector(n_pol, mu_hat[policy_idx], kernel, delta)
    mu_hat_claims <- exp(gamma[1] + gamma[2]*age_pol)
    b_claims      <- mu_hat_claims / alpha_hat
    phi_vals      <- get_phi_vector(sev_mean, alpha_hat, b_claims, kernel, gammaL)
    z <- psi_vals * phi_vals
    
    L <- if(any(z>0)) max(-1/z[z>0]) else -Inf
    U <- if(any(z<0)) min(-1/z[z<0]) else Inf
    L <- L + .Machine$double.eps;  U <- U - .Machine$double.eps
    negll_omega <- function(w) { -sum(log(1 + w * z)) }
    omega <- optimize(negll_omega, interval=c(L,U))$minimum
    
    # Update severity GLM (policy level)
    negll_gamma <- function(g) {
      mu_hat    <- exp(g[1] + g[2]*age_pol)
      b_hat     <- mu_hat / alpha_hat
      phi_ij <- get_phi_vector(sev_mean, alpha_hat, b_hat, kernel, gammaL)
      W      <- 1 + omega * psi_vals * phi_ij
      if(!all(is.finite(W)) || any(W <= 0)) return(1e10)
      ll1 <- sum(dgamma(sev_mean, shape=alpha_hat, scale=b_hat, log=TRUE))
      ll2 <- sum(log(W))
      -(ll1 + ll2)
    }
    optg  <- optim(gamma, negll_gamma, method="BFGS")
    gamma <- optg$par
    
    # Refit NB regression (frequency GLM)
    freq_glm <- glm.nb(count ~ age, data=nb_df)
    mu_hat <- predict(freq_glm, type="response")
    
    if(max(abs(omega-omega_old), abs(gamma-gamma_old)) < tol) {
      if(verbose > 0) cat("Converged! Breaking.\n")
      break
    }
    if(verbose > 0) cat("NB-Gamma Sarmanov omega:", round(omega,4), "\n")
  }
  
  # Return results in a way matching your NN output
  list(
    freq_glm = freq_glm,
    sev_glm = gamma,
    alpha_hat = alpha_hat,
    omega = omega,
    r_hat = r_hat 
  )
  
}


fit_poisson_lognormal_glm_sarmanov <- function(
    X, freq, sev_list,
    kernel = "standardized",      # or "exponential"
    delta = 0.5,                  # for exponential psi, if needed
    gammaL = 0.5,                 # for exponential phi
    max_iter = 50,
    tol = 1e-2,
    verbose = 2
) {
  age <- as.numeric(X[,1])
  poisson_df <- data.frame(age = age, count = freq)
  sev        <- unlist(sev_list)
  policy_idx <- which(freq > 0)
  age_pol    <- age[policy_idx]
  n_pol      <- freq[policy_idx]
  sev_mean   <- sapply(sev_list[policy_idx], mean)
  
  # Estimate lognormal sigma globally
  sigma_hat <- sd(log(sev))
  
  # For each policy, implied meanlog
  mu_implied <- log(sev_mean) - 0.5 * sigma_hat^2
  
  # Initial Poisson regression (frequency GLM)
  freq_glm <- glm(count ~ age, family=poisson(link="log"), data=poisson_df)
  lambda_hat <- predict(freq_glm, type="response")
  
  # Initial severity GLM (meanlog regression, policy-level)
  sev_df <- data.frame(age=age_pol, logsev=log(sev_mean))
  sev_glm <- lm(logsev ~ age, data=sev_df)
  beta <- coef(sev_glm)
  
  # --- Kernels ---
  phi_exp_lognormal <- function(x, mu, sigma, gammaL) exp(-gammaL * x) - exp(mu * (-gammaL) + 0.5 * sigma^2 * gammaL^2)
  phi_std_lognormal <- function(x, mu, sigma) (x - mu) / sigma
  get_phi_vector <- function(x, mu, sigma, kernel, gammaL) {
    if(kernel == "exponential") {
      phi_exp_lognormal(x, mu, sigma, gammaL)
    } else {
      phi_std_lognormal(x, mu, sigma)
    }
  }
  get_psi_vector <- function(n, lambda, kernel, delta) {
    if(kernel == "exponential") {
      exp(-delta * n) - exp(-delta * lambda)
    } else {
      (n - lambda) / sqrt(lambda)
    }
  }
  
  omega <- 0.1
  for(iter in 1:max_iter) {
    omega_old <- omega
    beta_old <- beta
    
    psi_vals <- get_psi_vector(n_pol, lambda_hat[policy_idx], kernel, delta)
    mu_hat_policies <- beta[1] + beta[2]*age_pol
    phi_vals <- get_phi_vector(log(sev_mean), mu_hat_policies, sigma_hat, kernel, gammaL)
    z <- psi_vals * phi_vals
    
    L <- if(any(z>0)) max(-1/z[z>0]) else -Inf
    U <- if(any(z<0)) min(-1/z[z<0]) else Inf
    L <- L + .Machine$double.eps;  U <- U - .Machine$double.eps
    negll_omega <- function(w) { -sum(log(1 + w * z)) }
    omega <- optimize(negll_omega, interval=c(L,U))$minimum
    
    # Update severity meanlog regression
    negll_beta <- function(b) {
      mu_hat <- b[1] + b[2]*age_pol
      phi_ij <- get_phi_vector(log(sev_mean), mu_hat, sigma_hat, kernel, gammaL)
      W      <- 1 + omega * psi_vals * phi_ij
      if(!all(is.finite(W)) || any(W <= 0)) return(1e10)
      ll1 <- sum(dnorm(log(sev_mean), mean=mu_hat, sd=sigma_hat, log=TRUE))
      ll2 <- sum(log(W))
      -(ll1 + ll2)
    }
    optb <- optim(beta, negll_beta, method="BFGS")
    beta <- optb$par
    
    # Update frequency GLM (no omega effect)
    freq_glm <- glm(count ~ age, family=poisson(link="log"), data=poisson_df)
    lambda_hat <- predict(freq_glm, type="response")
    
    if(max(abs(omega-omega_old), abs(beta-beta_old)) < tol) {
      if(verbose > 0) cat("Converged! Breaking.\n")
      break
    }
    if(verbose > 0) cat("Poisson-Lognormal Sarmanov omega:", round(omega,4), "\n")
  }
  
  # Return object for easy predictions
  list(
    freq_glm = freq_glm,
    sev_glm = beta,
    sigma_hat = sigma_hat,
    omega = omega
  )
}






fit_zip_lognormal_glm_sarmanov <- function(
    X, freq, sev_list,
    kernel = "standardized",
    delta = 0.5,
    gammaL = 0.5,
    max_iter = 50,
    tol = 1e-2,
    verbose = 2
) {
  library(pscl)
  age <- as.numeric(X[,1])
  zip_df <- data.frame(age=age, count=freq)
  sev    <- unlist(sev_list)
  policy_idx <- which(freq > 0)
  
  # Edge case: No nonzero frequency
  if(length(policy_idx) == 0) {
    warning("No nonzero frequency policies. Returning defaults.")
    return(list(
      freq_glm = NULL,
      sev_glm = c(NA, NA),
      sigma_hat = NA,
      omega = 0,
      pi_hat = 1,
      lambda_hat = rep(0, length(freq))
    ))
  }
  
  age_pol    <- age[policy_idx]
  n_pol      <- freq[policy_idx]
  sev_mean   <- sapply(sev_list[policy_idx], mean)
  
  # Fit ZIP globally (intercept only)
  zip_fit <- tryCatch(pscl::zeroinfl(count ~ 1 | 1, data=zip_df, dist="poisson"), error=function(e) NULL)
  if (is.null(zip_fit)) {
    warning("ZIP fit failed. Returning defaults.")
    return(list(
      freq_glm = NULL,
      sev_glm = c(NA, NA),
      sigma_hat = NA,
      omega = 0,
      pi_hat = 1,
      lambda_hat = rep(0, length(freq))
    ))
  }
  pi_hat <- plogis(coef(zip_fit)["zero_(Intercept)"])
  
  # Use Poisson GLM for lambda vs age (flexible for out-of-sample prediction)
  freq_glm <- glm(count ~ age, family=poisson, data=zip_df)
  lambda_hat <- predict(freq_glm, type="response")
  
  # Severity GLM (meanlog regression, policy-level)
  if (length(sev_mean) < 2 || sd(log(sev_mean)) == 0) {
    warning("Not enough or constant severity data. Using fallback regression.")
    sigma_hat <- sd(log(sev))
    beta <- c(mean(log(sev_mean)), 0)
  } else {
    sigma_hat <- sd(log(sev))
    sev_df <- data.frame(age=age_pol, logsev=log(sev_mean))
    sev_glm <- lm(logsev ~ age, data=sev_df)
    beta <- coef(sev_glm)
  }
  
  # Kernel functions
  phi_exp_lognormal <- function(x, mu, sigma, gammaL) exp(-gammaL * x) - exp(mu * (-gammaL) + 0.5 * sigma^2 * gammaL^2)
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
      exp(-delta * n) - exp(-delta * (1-pi)*lambda)
    } else {
      (n - (1-pi)*lambda) / sqrt((1-pi)*lambda + 1e-10) # epsilon for safety
    }
  }
  
  omega <- 0.1
  for(iter in 1:max_iter) {
    omega_old <- omega
    beta_old  <- beta
    
    psi_vals <- get_psi_vector(n_pol, lambda_hat[policy_idx], pi_hat, kernel, delta)
    mu_hat_policies <- beta[1] + beta[2]*age_pol
    phi_vals <- get_phi_vector(log(sev_mean), mu_hat_policies, sigma_hat, kernel, gammaL)
    z <- psi_vals * phi_vals
    finite_z <- z[is.finite(z)]
    
    # If all z=0, or all non-finite, set omega=0 and break
    if (length(finite_z) == 0 || all(finite_z == 0) || all(!is.finite(finite_z))) {
      omega <- 0
      break
    } else {
      L <- if(any(finite_z > 0)) max(-1/finite_z[finite_z > 0]) else -Inf
      U <- if(any(finite_z < 0)) min(-1/finite_z[finite_z < 0]) else Inf
      L <- L + .Machine$double.eps;  U <- U - .Machine$double.eps
      if(!is.finite(L) || !is.finite(U) || L >= U) {
        stop(sprintf("Bad omega optimization bounds: [%g, %g]", L, U))
      }
      negll_omega <- function(w) {
        W <- 1 + w * z
        if(any(W <= 0) || any(!is.finite(W))) return(Inf)
        -sum(log(W))
      }
      omega <- optimize(negll_omega, interval=c(L, U))$minimum
    }
    
    # Update severity meanlog regression
    negll_beta <- function(b) {
      mu_hat <- b[1] + b[2]*age_pol
      phi_ij <- get_phi_vector(log(sev_mean), mu_hat, sigma_hat, kernel, gammaL)
      W      <- 1 + omega * psi_vals * phi_ij
      if(!all(is.finite(W)) || any(W <= 0)) return(1e10)
      ll1 <- sum(dnorm(log(sev_mean), mean=mu_hat, sd=sigma_hat, log=TRUE))
      ll2 <- sum(log(W))
      -(ll1 + ll2)
    }
    if (all(is.na(beta))) beta <- c(0,0)
    optb <- tryCatch(optim(beta, negll_beta, method="BFGS"), error=function(e) list(par=beta))
    beta <- optb$par
    
    if(max(abs(omega-omega_old), abs(beta-beta_old)) < tol) {
      if(verbose > 0) cat("Converged! Breaking.\n")
      break
    }
    if(verbose > 0) cat("ZIP-Lognormal Sarmanov omega:", round(omega,4), "\n")
  }
  
  list(
    freq_glm = freq_glm,     # For prediction on new data!
    sev_glm = beta,          # meanlog coefficients (intercept, slope)
    sigma_hat = sigma_hat,
    omega = omega,
    pi_hat = pi_hat,
    lambda_hat = lambda_hat  # Fitted lambda for training data, for diagnostic
  )
}






fit_nb_lognormal_glm_sarmanov <- function(
    X, freq, sev_list,
    kernel = "standardized",
    delta = 0.5,
    gammaL = 0.5,
    max_iter = 50,
    tol = 1e-2,
    verbose = 2
) {
  require(MASS)
  age <- as.numeric(X[, 1])
  nb_df <- data.frame(age = age, count = freq)
  sev <- unlist(sev_list)
  policy_idx <- which(freq > 0)
  
  # --- Negative Binomial frequency fit ---
  freq_glm <- tryCatch(MASS::glm.nb(count ~ age, data = nb_df), error = function(e) NULL)
  if (!is.null(freq_glm)) {
    r_hat <- freq_glm$theta
    mu_hat <- as.vector(predict(freq_glm, type = "response"))
  } else {
    warning("NB regression failed! All downstream freq will be NA/0.")
    r_hat <- NA
    mu_hat <- rep(NA, length(age))
  }
  
  # --- Severity GLM (meanlog regression, policy-level) ---
  if (length(policy_idx) == 0) {
    warning("No nonzero frequency policies. Returning defaults.")
    return(list(
      freq_glm = freq_glm,
      sev_glm = c(NA, NA),
      sigma_hat = NA,
      omega = 0,
      r_hat = r_hat,
      mu_hat = mu_hat
    ))
  }
  
  age_pol <- age[policy_idx]
  n_pol <- freq[policy_idx]
  sev_mean <- sapply(sev_list[policy_idx], mean)
  
  if (length(sev_mean) < 2 || sd(log(sev_mean)) == 0) {
    warning("Not enough or constant severity data. Using fallback regression.")
    sigma_hat <- sd(log(sev))
    beta <- c(mean(log(sev_mean)), 0)
  } else {
    sigma_hat <- sd(log(sev))
    sev_df <- data.frame(age = age_pol, logsev = log(sev_mean))
    sev_glm <- lm(logsev ~ age, data = sev_df)
    beta <- coef(sev_glm)
  }
  
  # --- Sarmanov kernels ---
  phi_exp_lognormal <- function(x, mu, sigma, gammaL) exp(-gammaL * x) - exp(mu * (-gammaL) + 0.5 * sigma^2 * gammaL^2)
  phi_std_lognormal <- function(x, mu, sx) (x - mu) / sx
  get_phi_vector <- function(x, mu, sigma, kernel, gammaL) {
    if (kernel == "exponential") {
      phi_exp_lognormal(x, mu, sigma, gammaL)
    } else {
      phi_std_lognormal(x, mu, sigma)
    }
  }
  get_psi_vector <- function(n, freq_pred, kernel, delta) {
    if (kernel == "exponential") {
      exp(-delta * n) - exp(-delta * freq_pred)
    } else {
      (n - freq_pred) / sqrt(freq_pred + 1e-10)
    }
  }
  
  # --- Sarmanov alternating updates ---
  omega <- 0.1
  for (iter in 1:max_iter) {
    omega_old <- omega
    beta_old <- beta
    
    # Predict freq for each positive-freq policy
    freq_pred <- as.vector(predict(freq_glm, newdata = data.frame(age = age_pol), type = "response"))
    psi_vals <- get_psi_vector(n_pol, freq_pred, kernel, delta)
    mu_hat_policies <- beta[1] + beta[2] * age_pol
    phi_vals <- get_phi_vector(log(sev_mean), mu_hat_policies, sigma_hat, kernel, gammaL)
    z <- psi_vals * phi_vals
    finite_z <- z[is.finite(z)]
    
    # Sarmanov omega optimization
    if (length(finite_z) == 0 || all(finite_z == 0) || all(!is.finite(finite_z))) {
      omega <- 0
      break
    } else {
      L <- if (any(finite_z > 0)) max(-1 / finite_z[finite_z > 0]) else -Inf
      U <- if (any(finite_z < 0)) min(-1 / finite_z[finite_z < 0]) else Inf
      L <- L + .Machine$double.eps; U <- U - .Machine$double.eps
      if (!is.finite(L) || !is.finite(U) || L >= U) {
        stop(sprintf("Bad omega optimization bounds: [%g, %g]", L, U))
      }
      negll_omega <- function(w) {
        W <- 1 + w * z
        if (any(W <= 0) || any(!is.finite(W))) return(Inf)
        -sum(log(W))
      }
      omega <- optimize(negll_omega, interval = c(L, U))$minimum
    }
    
    # Update severity meanlog regression
    negll_beta <- function(b) {
      mu_hat <- b[1] + b[2] * age_pol
      phi_ij <- get_phi_vector(log(sev_mean), mu_hat, sigma_hat, kernel, gammaL)
      W <- 1 + omega * psi_vals * phi_ij
      if (!all(is.finite(W)) || any(W <= 0)) return(1e10)
      ll1 <- sum(dnorm(log(sev_mean), mean = mu_hat, sd = sigma_hat, log = TRUE))
      ll2 <- sum(log(W))
      -(ll1 + ll2)
    }
    if (all(is.na(beta))) beta <- c(0, 0)
    optb <- tryCatch(optim(beta, negll_beta, method = "BFGS"), error = function(e) list(par = beta))
    beta <- optb$par
    
    if (max(abs(omega - omega_old), abs(beta - beta_old)) < tol) {
      if (verbose > 0) cat("Converged! Breaking.\n")
      break
    }
    if (verbose > 0) cat("NB-Lognormal Sarmanov omega:", round(omega, 4), "\n")
  }
  
  list(
    freq_glm = freq_glm,   # Fitted NB regression (call predict for new X)
    sev_glm = beta,        # meanlog coefficients (intercept, slope)
    sigma_hat = sigma_hat, # Lognormal sigma for severity
    omega = omega,         # Sarmanov fitted dependence
    r_hat = r_hat,         # NB overdispersion
    mu_hat = mu_hat        # Fitted NB mean for training set (can be recomputed for new data)
  )
}







set.seed(369)
M <- 10000         
alpha0 <- 2.87        
omega0 <- 0.3      

age <- runif(M, 20, 60)
lambda_true <- 0.003*(age-30)^2 + exp(0.05*(age-40) - 0.008*(age-40)^2 + log(3)) + 0.6
mu_true     <- 0.08*abs(age-40) + exp(0.3*sqrt(age) - 0.12*(age-40)^2 + log(0.5)) + 2
beta_true   <- mu_true / alpha0

psi_fun <- function(n, lam)      (n - lam)/sqrt(lam)
phi_fun <- function(x, mu, sx)   (x - mu)/sx

freq <- integer(M)
sev_list <- vector("list", M)

for(i in seq_len(M)) {
  lam <- lambda_true[i]
  n_i <- rpois(1, lam)
  freq[i] <- n_i
  if(n_i > 0L) {
    psi_i <- psi_fun(n_i, lam)
    draw_one <- function() {
      repeat {
        x_prop <- rgamma(1, shape=alpha0, scale=beta_true[i])
        w      <- 1 + omega0 * psi_i * phi_fun(x_prop, mu_true[i], sqrt(alpha0)*beta_true[i])
        if (w>0 && runif(1) < w) return(x_prop)
      }
    }
    sev_list[[i]] <- replicate(n_i, draw_one())
  } else {
    sev_list[[i]] <- numeric(0)
  }
}
sev     <- unlist(sev_list)
pid     <- rep(seq_along(sev_list), lengths(sev_list))
age_rep <- age[pid]
X       <- matrix(age, ncol=1)



# --- List of model fit functions, run all six models ---
fit_functions <- list(
  poisson_gamma   = fit_poisson_gamma_glm_sarmanov,
  nb_gamma        = fit_nb_gamma_glm_sarmanov,
  zip_gamma       = fit_zip_gamma_glm_sarmanov,
  poisson_lognorm = fit_poisson_lognormal_glm_sarmanov,
  nb_lognorm      = fit_nb_lognormal_glm_sarmanov,
  zip_lognorm     = fit_zip_lognormal_glm_sarmanov
)

results <- list()
for(model in names(fit_functions)) {
  cat("\n--- Running", model, "---\n")
  # Call the fit function, adjust arguments as needed (e.g., kernel="standardized")
  results[[model]] <- fit_functions[[model]](
    X, freq, sev_list, kernel="exponential", verbose=1
  )
}
model_names <- c(
  "poisson_gamma", "nb_gamma", "zip_gamma",
  "poisson_lognorm", "nb_lognorm", "zip_lognorm"
)

# ---plot ---
# True lambda on grid
age_grid <- seq(20, 60, by=1)

# Define representative models (use your gamma models as archetypes)
model_families <- c(
  "poisson_gamma",    # for Poisson freq
  "nb_gamma",         # for NB freq
  "zip_gamma"         # for ZIP freq
)
colors <- c("purple", "red", "blue")
labels <- c("Poisson", "NB", "ZIP")

lambda_true_grid <- 0.003*(age_grid-30)^2 + exp(0.05*(age_grid-40) - 0.008*(age_grid-40)^2 + log(3)) + 0.6


# Plot true lambda
plot(age_grid, lambda_true_grid, type="l", col="black", lwd=2,
     ylim=range(lambda_true_grid), main="Frequency Fitting", ylab="Lambda", xlab="Age")

# Overlay each model family
for (i in seq_along(model_families)) {
  model <- model_families[i]
  res <- results[[model]]
  if (!is.null(res$freq_glm)) {
    pred_lambda <- predict(res$freq_glm, newdata=data.frame(age=age_grid), type="response")
  } else if (!is.null(res$lambda_hat)) {
    pred_lambda <- res$lambda_hat[match(age_grid, sort(unique(as.integer(age_grid))))]
    if (is.null(pred_lambda) || length(pred_lambda) != length(age_grid)) {
      pred_lambda <- rep(NA, length(age_grid))
    }
  } else {
    pred_lambda <- rep(NA, length(age_grid))
  }
  if (!is.null(res$pi_hat) && !is.null(pred_lambda)) {
    pred_lambda <- (1 - res$pi_hat) * pred_lambda
  }
  if (!is.null(pred_lambda) && length(pred_lambda) == length(age_grid)) {
    lines(age_grid, pred_lambda, col=colors[i], lwd=2, lty=i+1)
  } else {
    warning(sprintf("Model %s: could not plot lambda (wrong length)", model))
  }
}

legend("topleft", legend=c("True", labels),
       col=c("black", colors[seq_along(labels)]), lwd=2, lty=1:(length(labels)+1))



# True mean severity (Gamma mean or Lognormal mean)
mu_true_grid     <- 0.08*abs(age_grid-40) + exp(0.3*sqrt(age_grid) - 0.12*(age_grid-40)^2 + log(0.5)) + 2
alpha0           <- 2.87  # Or your true value
beta_true_grid   <- mu_true_grid / alpha0
meanlog_true     <- log(mu_true_grid) - 0.5 * (ifelse(any(grepl("lognorm", model_names)), 0.4^2, 0)) # if you want lognormal

# Compute true mean severity (mu_true_grid) and gamma scale (beta_true_grid)
# mu_true_grid     <- ...
# beta_true_grid   <- mu_true_grid / alpha0

colors <- c("purple", "red", "blue", "orange", "brown", "green")
labels <- c(
  "Poisson-Gamma", "NB-Gamma", "ZIP-Gamma",
  "Poisson-Lognormal", "NB-Lognormal", "ZIP-Lognormal"
)


# Plot TRUE mean severity
plot(age_grid, mu_true_grid, type="l", col="black", lwd=2,
     ylim=range(mu_true_grid), main="Severity Fitting", ylab="Mean Severity per Policy", xlab="Age")

for (i in seq_along(model_names)) {
  model <- model_names[i]
  res <- results[[model]]
  pred_sev <- NULL
  # Gamma severity: need fitted beta parameter
  if (grepl("gamma", model)) {
    # Predict beta (scale) as function of age using fitted severity GLM
    if (!is.null(res$sev_glm) && length(res$sev_glm) == 2) {
      # Recall: For gamma, mean = alpha * beta; beta = mean / alpha
      pred_mean <- exp(res$sev_glm[1] + res$sev_glm[2] * age_grid)
      beta <- if (!is.null(res$alpha_hat)) pred_mean / res$alpha_hat else rep(NA, length(age_grid))
      pred_sev <- pred_mean  # Plot the mean (alpha*beta), not just beta
    }
  } else if (grepl("lognorm", model)) {
    # Lognormal: mean = exp(mu + 0.5 * sigma^2)
    if (!is.null(res$sev_glm) && length(res$sev_glm) == 2 && !is.null(res$sigma_hat)) {
      mu_pred <- res$sev_glm[1] + res$sev_glm[2] * age_grid
      pred_sev <- exp(mu_pred + 0.5 * res$sigma_hat^2)
    }
  }
  if (!is.null(pred_sev) && length(pred_sev) == length(age_grid)) {
    lines(age_grid, pred_sev, col=colors[i], lwd=2, lty=i+1)
  } else {
    warning(sprintf("Model %s: could not plot severity (wrong length)", model))
  }
}

legend("topleft", legend=c("True", labels), col=c("black", colors), lwd=2, lty=1:(length(labels)+1))




# ---Aggregate Loss---


simulate_aggregate_loss_glm <- function(fit, X_val, M_val,
                                        model_type = c("poisson", "zip", "nb"),
                                        sev_type   = c("gamma", "lognormal"),
                                        pi_hat = NULL) {
  age_val <- as.numeric(X_val[, 1])
  # --- Frequency prediction ---
  if (model_type == "poisson") {
    lambda_pred <- as.vector(predict(fit$freq_glm, newdata=data.frame(age=age_val), type="response"))
    freq_pred <- rpois(M_val, lambda_pred)
  } else if (model_type == "zip") {
    # ZIP: mean λ * (1 - pi_hat)
    if (is.null(pi_hat)) pi_hat <- fit$pi_hat
    lambda_pred <- as.vector(predict(fit$freq_glm, newdata=data.frame(age=age_val), type="response"))
    freq_pred <- rbinom(M_val, 1, 1-pi_hat) * rpois(M_val, lambda_pred)
  } else if (model_type == "nb") {
    # Negative Binomial: need size (r_hat) and mean (mu_hat), but use GLM for μ
    r_hat <- if(!is.null(fit$r_hat)) fit$r_hat else 1
    mu_pred <- as.vector(predict(fit$freq_glm, newdata=data.frame(age=age_val), type="response"))
    # R's rnbinom uses mu and size, prob is computed from mu/size
    freq_pred <- rnbinom(M_val, size=r_hat, mu=mu_pred)
  } else {
    stop("Unknown model_type")
  }
  
  # --- Severity prediction ---
  age_pol_idx <- which(freq_pred > 0)
  age_val_pol <- age_val[age_pol_idx]
  n_pol       <- freq_pred[age_pol_idx]
  
  if (sev_type == "gamma") {
    # meanlog = β₀ + β₁*age, mean = exp(meanlog), scale = mean/alpha
    alpha_hat <- if(!is.null(fit$alpha_hat)) fit$alpha_hat else 2  # fallback
    mu_pred <- fit$sev_glm[1] + fit$sev_glm[2] * age_val_pol
    mean_pred <- exp(mu_pred)
    beta_pred <- mean_pred / alpha_hat
    sev_draws <- lapply(seq_along(age_pol_idx), function(j)
      rgamma(n_pol[j], shape=alpha_hat, scale=beta_pred[j]))
  } else if (sev_type == "lognormal") {
    # meanlog = β₀ + β₁*age
    sigma_hat <- fit$sigma_hat
    mu_pred <- fit$sev_glm[1] + fit$sev_glm[2] * age_val_pol
    sev_draws <- lapply(seq_along(age_pol_idx), function(j)
      rlnorm(n_pol[j], meanlog=mu_pred[j], sdlog=sigma_hat))
  } else {
    stop("Unknown sev_type")
  }
  
  # Combine aggregate losses
  agg_loss <- numeric(M_val)
  agg_loss[] <- 0
  for (j in seq_along(age_pol_idx)) {
    i <- age_pol_idx[j]
    agg_loss[i] <- sum(sev_draws[[j]])
  }
  agg_loss
}


set.seed(888)
M_val <- 10000
age_val <- runif(M_val, 20, 60)
X_val <- matrix(age_val, ncol = 1)

# Simulate true aggregate loss as before (reuse your simulation code)
agg_loss_true <- sapply(sev_list_val, sum)

# Each fitted model (assumes your results list is called 'results')
agg_pg <- simulate_aggregate_loss_glm(results$poisson_gamma,   X_val, M_val, "poisson", "gamma")
agg_zg <- simulate_aggregate_loss_glm(results$zip_gamma,       X_val, M_val, "zip",     "gamma")
agg_ng <- simulate_aggregate_loss_glm(results$nb_gamma,        X_val, M_val, "nb",      "gamma")
agg_pl <- simulate_aggregate_loss_glm(results$poisson_lognorm, X_val, M_val, "poisson", "lognormal")
agg_zl <- simulate_aggregate_loss_glm(results$zip_lognorm,     X_val, M_val, "zip",     "lognormal")
agg_nl <- simulate_aggregate_loss_glm(results$nb_lognorm,      X_val, M_val, "nb",      "lognormal")


# ---- Set parameters for true data----
alpha0 <- 2.87
omega0 <- 0.3

lambda_true_val <- 0.003*(age_val-30)^2 + exp(0.05*(age_val-40) - 0.008*(age_val-40)^2 + log(3)) + 0.6
mu_true_val     <- 0.08*abs(age_val-40) + exp(0.3*sqrt(age_val) - 0.12*(age_val-40)^2 + log(0.5)) + 2
beta_true_val   <- mu_true_val / alpha0

psi_fun <- function(n, lam)      (n - lam)/sqrt(lam)
phi_fun <- function(x, mu, sx)   (x - mu)/sx

freq_val     <- integer(M_val)
sev_list_val <- vector("list", M_val)

for(i in seq_len(M_val)) {
  lam <- lambda_true_val[i]
  n_i <- rpois(1, lam)
  freq_val[i] <- n_i
  if(n_i > 0L) {
    psi_i <- psi_fun(n_i, lam)
    draw_one <- function() {
      repeat {
        x_prop <- rgamma(1, shape=alpha0, scale=beta_true_val[i])
        w      <- 1 + omega0 * psi_i * phi_fun(x_prop, mu_true_val[i], sqrt(alpha0)*beta_true_val[i])
        if (w>0 && runif(1) < w) return(x_prop)
      }
    }
    sev_list_val[[i]] <- replicate(n_i, draw_one())
  } else {
    sev_list_val[[i]] <- numeric(0)
  }
}

agg_loss_true <- sapply(sev_list_val, sum)




library(moments)

get_loss_stats <- function(loss_vec) {
  q <- quantile(loss_vec, probs = c(0.05, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999))
  c(
    Mean = mean(loss_vec),
    Std  = sd(loss_vec),
    Skew = skewness(loss_vec),
    Kurtosis = kurtosis(loss_vec),
    `5%` = q[1],
    `50%` = q[2],
    `75%` = q[3],
    `90%` = q[4],
    `95%` = q[5],
    `99%` = q[6],
    `99.9%` = q[7]
  )
}

agg_stats <- rbind(
  Truth               = get_loss_stats(agg_loss_true),
  `Poisson-Gamma`     = get_loss_stats(agg_pg),
  `ZIP-Gamma`         = get_loss_stats(agg_zg),
  `NB-Gamma`          = get_loss_stats(agg_ng),
  `Poisson-Lognormal` = get_loss_stats(agg_pl),
  `ZIP-Lognormal`     = get_loss_stats(agg_zl),
  `NB-Lognormal`      = get_loss_stats(agg_nl)
)
print(round(agg_stats, 3))

print(round(agg_stats, 3))


get_policy_predictions_glm <- function(fit, ages, dist, sevtype) {
  # Frequency prediction
  if (dist == "poisson") {
    freq_pred <- as.vector(predict(fit$freq_glm, newdata = data.frame(age = ages), type = "response"))
  } else if (dist == "zip") {
    lambda_pred <- as.vector(predict(fit$freq_glm, newdata = data.frame(age = ages), type = "response"))
    pi_hat <- if (!is.null(fit$pi_hat)) fit$pi_hat else 0
    freq_pred <- (1 - pi_hat) * lambda_pred
  } else if (dist == "nb") {
    mu_pred <- as.vector(predict(fit$freq_glm, newdata = data.frame(age = ages), type = "response"))
    freq_pred <- mu_pred
  } else {
    stop("Unknown frequency dist")
  }
  
  # Severity prediction
  if (sevtype == "gamma") {
    # mean = exp(β₀ + β₁ * age), scale = mean / alpha
    mu_pred <- fit$sev_glm[1] + fit$sev_glm[2] * ages
    mean_pred <- exp(mu_pred)
    sev_pred <- mean_pred / fit$alpha_hat  # beta
  } else if (sevtype == "lognormal") {
    mu_pred <- fit$sev_glm[1] + fit$sev_glm[2] * ages
    sev_pred <- mu_pred
  } else {
    stop("Unknown severity dist")
  }
  list(freq_pred = freq_pred, sev_pred = sev_pred)
}



run_all_glm_models_and_compare <- function(X, freq, sev_list, fits) {
  # Model names for display and corresponding keys in fits list
  model_names <- c("Poisson-Gamma", "ZIP-Gamma", "NB-Gamma",
                   "Poisson-Lognormal", "ZIP-Lognormal", "NB-Lognormal")
  model_keys  <- c("poisson_gamma", "zip_gamma", "nb_gamma",
                   "poisson_lognorm", "zip_lognorm", "nb_lognorm")
  dists       <- c("poisson", "zip", "nb", "poisson", "zip", "nb")
  sevs        <- c("gamma", "gamma", "gamma", "lognormal", "lognormal", "lognormal")
  
  # Observed values from data
  obs_freq     <- freq
  obs_sev_mean <- sapply(sev_list, function(x) if(length(x) > 0) mean(x) else NA_real_)
  obs_agg      <- sapply(sev_list, sum)
  
  # RMSE utility
  rmse <- function(est, true) {
    idx <- which(!is.na(est) & !is.na(true) & !is.nan(est) & !is.nan(true))
    if (length(idx) == 0) return(NA_real_)
    sqrt(mean((est[idx] - true[idx])^2))
  }
  
  # Prediction function: returns expected frequency and severity per policy
  get_policy_predictions_glm <- function(fit, ages, dist, sevtype) {
    # Predict frequency
    if (dist == "poisson") {
      freq_pred <- as.vector(predict(fit$freq_glm, newdata = data.frame(age = ages), type = "response"))
    } else if (dist == "zip") {
      lambda_pred <- as.vector(predict(fit$freq_glm, newdata = data.frame(age = ages), type = "response"))
      pi_hat <- if (!is.null(fit$pi_hat)) fit$pi_hat else 0
      freq_pred <- (1 - pi_hat) * lambda_pred
    } else if (dist == "nb") {
      mu_pred <- as.vector(predict(fit$freq_glm, newdata = data.frame(age = ages), type = "response"))
      freq_pred <- mu_pred
    } else {
      stop("Unknown frequency distribution")
    }
    # Predict severity (either gamma mean or lognormal meanlog)
    if (sevtype == "gamma") {
      beta_pred <- fit$sev_glm[1] + fit$sev_glm[2] * ages
      sev_pred <- beta_pred*fit$alpha_hat
    } else if (sevtype == "lognormal") {
      coefs <- if (inherits(fit$sev_glm, "lm")) fit$sev_glm$coefficients else fit$sev_glm
      mu_pred <- coefs[1] + coefs[2] * ages
      sev_pred <- exp(mu_pred + 0.5 * fit$sigma_hat^2)
    } else {
      stop("Unknown severity distribution")
    }
    list(freq_pred = freq_pred, sev_pred = sev_pred)
  }
  
  
  

  
  # Arrays to hold statistics for each model
  rmse_freq <- numeric(6)
  rmse_sev  <- numeric(6)
  rmse_agg  <- numeric(6)
  logliks   <- numeric(6)
  omegas    <- numeric(6)
  
  # Loop through each model, compute statistics, print debug info
  for (j in seq_along(model_names)) {
    key <- model_keys[j]
    fit <- fits[[key]]
    pred <- get_policy_predictions_glm(fit, X[,1], dists[j], sevs[j])
    rmse_freq[j] <- rmse(pred$freq_pred, obs_freq)
    idx_nonzero <- which(obs_freq > 0 & !is.na(obs_sev_mean))
    rmse_sev[j] <- rmse(pred$sev_pred[idx_nonzero], obs_sev_mean[idx_nonzero])
    pred_agg <- pred$freq_pred * pred$sev_pred
    rmse_agg[j] <- rmse(pred_agg, obs_agg)
    omegas[j] <- if (!is.null(fit$omega)) fit$omega else NA
    }
  
  # Assemble results in a tibble
  stats <- tibble::tibble(
    Model          = model_names,
    Omega          = omegas,
    RMSE_Frequency = rmse_freq,
    RMSE_Severity  = rmse_sev,
    RMSE_Aggregate = rmse_agg,
  )
  return(list(
    stats_table = stats,
    fits = fits
  ))
}



# Fit all models
fit_functions <- list(
  poisson_gamma   = fit_poisson_gamma_glm_sarmanov,
  nb_gamma        = fit_nb_gamma_glm_sarmanov,
  zip_gamma       = fit_zip_gamma_glm_sarmanov,
  poisson_lognorm = fit_poisson_lognormal_glm_sarmanov,
  nb_lognorm      = fit_nb_lognormal_glm_sarmanov,
  zip_lognorm     = fit_zip_lognormal_glm_sarmanov
)
results <- lapply(fit_functions, function(f) f(X, freq, sev_list, kernel="exponential"))

# Comparison table
compare_results <- run_all_glm_models_and_compare(X_val, freq_val, sev_list_val, results)
print(compare_results$stats_table)

# If validation set
# stats_val <- evaluate_models_on_validation(results, X_val, freq_val, sev_list_val)
# print(stats_val)



compute_implied_corr <- function(results, freq, sev_list, delta=0.5, gammaL=0.5) {
  # Per-policy frequency and all-claim severity vectors
  N_vec <- freq
  X_vec <- unlist(sev_list)
  M <- length(N_vec)
  
  # Kernels (exponential, as used in your models)
  psiN_raw <- exp(-delta * N_vec)
  C_N <- mean(psiN_raw)
  psiN <- psiN_raw - C_N
  
  phiX_raw <- exp(-gammaL * X_vec)
  C_X <- mean(phiX_raw)
  phiX <- phiX_raw - C_X
  
  # Empirical means
  ENpsiN   <- mean(N_vec * psiN)             # policy-level
  EXphiX   <- mean(X_vec * phiX)             # claim-level
  
  # Sample standard deviations
  sigma_N <- sd(N_vec)
  sigma_X <- sd(X_vec)
  
  model_keys <- c(
    "poisson_gamma", "nb_gamma", "zip_gamma",
    "poisson_lognorm", "nb_lognorm", "zip_lognorm"
  )
  
  implied_corrs <- setNames(numeric(length(model_keys)), model_keys)
  
  for (key in model_keys) {
    mod <- results[[key]]
    omega_hat <- mod$omega
    implied_corrs[key] <- omega_hat * ENpsiN * EXphiX / (sigma_N * sigma_X)
  }
  implied_corrs
}

# --- Example usage (assuming all objects already exist in your session) ---
# freq      : integer vector of per-policy frequencies
# sev_list  : list of claim vectors (as in your simulation)
# results   : your list of fitted model results

implied_rhos <- compute_implied_corr(results, freq, sev_list, delta=0.5, gammaL=0.5)
print(implied_rhos)



