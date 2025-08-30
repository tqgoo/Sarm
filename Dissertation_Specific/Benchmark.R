# --- Fitting Functions: pure GLM, no dependence ---

fit_poisson_gamma_glm_implied <- function(X, freq, sev_list) {
  age <- as.numeric(X[,1])
  # Poisson freq implied param GLM
  freq_glm <- glm(freq ~ age, family=poisson)
  lambda_hat <- predict(freq_glm, type="response")
  # Severity: implied β
  idx <- which(freq > 0)
  sev_mean <- sapply(sev_list[idx], mean)
  sev <- unlist(sev_list)
  alpha_hat <- MASS::fitdistr(sev, "gamma")$estimate["shape"]
  beta <- sev_mean / alpha_hat
  # Only fit for beta > 0
  good_idx <- which(beta > 0)
  sev_df <- data.frame(age = age[idx][good_idx], beta = beta[good_idx])
  sev_glm <- glm(beta ~ age, family=Gamma(link="log"), data=sev_df)
  list(freq_glm=freq_glm, sev_glm=sev_glm, alpha_hat=alpha_hat)
}



fit_zip_gamma_glm_implied <- function(X, freq, sev_list) {
  require(pscl); require(MASS)
  age <- as.numeric(X[,1])
  
  # Estimate ZIP π globally (intercept only)
  fit_zip <- pscl::zeroinfl(freq ~ 1 | 1, dist="poisson")
  pi_hat <- plogis(coef(fit_zip)["zero_(Intercept)"])
  
  # Implied Poisson mean for each policy: λ_j = freq_j / (1 - pi_hat)
  lambda_implied <- freq/(1-pi_hat)
  # Only positive, finite lambda_implied
  good_lambda <- which(lambda_implied > 0 & is.finite(lambda_implied))
  if (length(good_lambda) < 2) stop("Not enough positive lambda_implied for GLM.")
  df_freq <- data.frame(lambda_implied = lambda_implied[good_lambda], age = age[good_lambda])
  freq_glm <- glm(lambda_implied ~ age, data = df_freq, family=Gamma(link="log"))
  
  # Gamma shape parameter (global)
  sev <- unlist(sev_list)
  alpha_hat <- MASS::fitdistr(sev, "gamma")$estimate["shape"]
  
  # Implied beta_j for policies with freq > 0
  idx <- which(freq > 0)
  sev_mean <- sapply(sev_list[idx], mean)
  beta_implied <- sev_mean / alpha_hat
  # Only keep positive beta_implied
  good_beta <- which(beta_implied > 0 & is.finite(beta_implied))
  if (length(good_beta) < 2) stop("Not enough positive beta_implied for severity GLM.")
  df_sev <- data.frame(beta_implied = beta_implied[good_beta], age = age[idx][good_beta])
  sev_glm <- glm(beta_implied ~ age, data = df_sev, family=Gamma(link="log"))
  
  list(freq_glm=freq_glm, sev_glm=sev_glm, alpha_hat=alpha_hat, pi_hat=pi_hat)
}



fit_nb_gamma_glm_implied <- function(X, freq, sev_list) {
  require(MASS)
  age <- as.numeric(X[,1])
  
  # Fit NB globally (intercept only) to get r_hat
  nb_fit <- glm.nb(freq ~ 1)
  r_hat <- nb_fit$theta
  
  # Implied mean and p_j for each policy
  mu_implied <- freq
  p_implied <- r_hat / (r_hat + mu_implied)
  good_p <- which(p_implied > 0 & p_implied < 1 & is.finite(p_implied))
  if (length(good_p) < 2) stop("Not enough valid p_implied for NB frequency GLM.")
  df_freq <- data.frame(p_implied = p_implied[good_p], age = age[good_p])
  freq_glm <- glm(p_implied ~ age, data = df_freq, family=binomial(link="logit"))
  
  # Gamma shape parameter (global)
  sev <- unlist(sev_list)
  alpha_hat <- MASS::fitdistr(sev[sev > 0], "gamma")$estimate["shape"]
  
  # Severity implied beta_j
  idx <- which(freq > 0)
  sev_mean <- sapply(sev_list[idx], mean)
  beta_implied <- sev_mean / alpha_hat
  good_beta <- which(beta_implied > 0 & is.finite(beta_implied))
  if (length(good_beta) < 2) stop("Not enough positive beta_implied for severity GLM.")
  df_sev <- data.frame(beta_implied = beta_implied[good_beta], age = age[idx][good_beta])
  sev_glm <- glm(beta_implied ~ age, data = df_sev, family=Gamma(link="log"))
  
  list(freq_glm=freq_glm, sev_glm=sev_glm, alpha_hat=alpha_hat, r_hat=r_hat)
}

fit_poisson_lognorm_glm_implied <- function(X, freq, sev_list) {
  age <- as.numeric(X[,1])
  # Poisson: λ_j = freq_j
  lambda_implied <- freq
  good_lambda <- which(lambda_implied > 0 & is.finite(lambda_implied))
  if (length(good_lambda) < 2) stop("Not enough positive lambda_implied for GLM.")
  df_freq <- data.frame(lambda_implied = lambda_implied[good_lambda], age = age[good_lambda])
  freq_glm <- glm(lambda_implied ~ age, data = df_freq, family=Gamma(link="log"))
  
  # Lognormal sigma (global)
  sev <- unlist(sev_list)
  sigma_hat <- sd(log(sev[sev > 0]))
  
  # Implied mu_j per policy (for policies with freq > 0)
  idx <- which(freq > 0)
  sev_mean <- sapply(sev_list[idx], mean)
  mu_implied <- log(sev_mean) - 0.5 * sigma_hat^2
  good_mu <- which(is.finite(mu_implied))
  if (length(good_mu) < 2) stop("Not enough valid mu_implied for severity GLM.")
  df_sev <- data.frame(mu_implied = mu_implied[good_mu], age = age[idx][good_mu])
  sev_glm <- lm(mu_implied ~ age, data = df_sev)
  
  list(freq_glm=freq_glm, sev_glm=sev_glm, sigma_hat=sigma_hat)
}

fit_zip_lognorm_glm_implied <- function(X, freq, sev_list) {
  require(pscl)
  age <- as.numeric(X[,1])
  fit_zip <- pscl::zeroinfl(freq ~ 1 | 1, dist="poisson")
  pi_hat <- plogis(coef(fit_zip)["zero_(Intercept)"])
  lambda_implied <- freq/(1-pi_hat)
  good_lambda <- which(lambda_implied > 0 & is.finite(lambda_implied))
  if (length(good_lambda) < 2) stop("Not enough positive lambda_implied for GLM.")
  df_freq <- data.frame(lambda_implied = lambda_implied[good_lambda], age = age[good_lambda])
  freq_glm <- glm(lambda_implied ~ age, data = df_freq, family=Gamma(link="log"))
  
  sev <- unlist(sev_list)
  sigma_hat <- sd(log(sev[sev > 0]))
  idx <- which(freq > 0)
  sev_mean <- sapply(sev_list[idx], mean)
  mu_implied <- log(sev_mean) - 0.5 * sigma_hat^2
  good_mu <- which(is.finite(mu_implied))
  if (length(good_mu) < 2) stop("Not enough valid mu_implied for severity GLM.")
  df_sev <- data.frame(mu_implied = mu_implied[good_mu], age = age[idx][good_mu])
  sev_glm <- lm(mu_implied ~ age, data = df_sev)
  
  list(freq_glm=freq_glm, sev_glm=sev_glm, sigma_hat=sigma_hat, pi_hat=pi_hat)
}

fit_nb_lognorm_glm_implied <- function(X, freq, sev_list) {
  require(MASS)
  age <- as.numeric(X[,1])
  nb_fit <- glm.nb(freq ~ 1)
  r_hat <- nb_fit$theta
  mu_implied <- freq
  p_implied <- r_hat / (r_hat + mu_implied)
  good_p <- which(p_implied > 0 & p_implied < 1 & is.finite(p_implied))
  if (length(good_p) < 2) stop("Not enough valid p_implied for NB frequency GLM.")
  df_freq <- data.frame(p_implied = p_implied[good_p], age = age[good_p])
  freq_glm <- glm(p_implied ~ age, data = df_freq, family=binomial(link="logit"))
  
  sev <- unlist(sev_list)
  sigma_hat <- sd(log(sev[sev > 0]))
  idx <- which(freq > 0)
  sev_mean <- sapply(sev_list[idx], mean)
  mu_implied <- log(sev_mean) - 0.5 * sigma_hat^2
  good_mu <- which(is.finite(mu_implied))
  if (length(good_mu) < 2) stop("Not enough valid mu_implied for severity GLM.")
  df_sev <- data.frame(mu_implied = mu_implied[good_mu], age = age[idx][good_mu])
  sev_glm <- lm(mu_implied ~ age, data = df_sev)
  
  list(freq_glm=freq_glm, sev_glm=sev_glm, sigma_hat=sigma_hat, r_hat=r_hat)
}





fit_all_glm_models_implied <- function(X, freq, sev_list) {
  list(
    poisson_gamma   = fit_poisson_gamma_glm_implied(X, freq, sev_list),
    zip_gamma       = fit_zip_gamma_glm_implied(X, freq, sev_list),
    nb_gamma        = fit_nb_gamma_glm_implied(X, freq, sev_list),
    poisson_lognorm = fit_poisson_lognorm_glm_implied(X, freq, sev_list),
    zip_lognorm     = fit_zip_lognorm_glm_implied(X, freq, sev_list),
    nb_lognorm      = fit_nb_lognorm_glm_implied(X, freq, sev_list)
  )
}


get_policy_predictions_glm_implied <- function(fit, ages, dist, sevtype) {
  # Frequency prediction
  if (dist == "poisson") {
    # λ_j regression
    lambda_pred <- as.vector(predict(fit$freq_glm, newdata=data.frame(age=ages), type="response"))
    freq_pred <- lambda_pred
  } else if (dist == "zip") {
    # λ_j regression, then E[N] = (1 - pi_hat) * lambda_j
    lambda_pred <- as.vector(predict(fit$freq_glm, newdata=data.frame(age=ages), type="response"))
    pi_hat <- if (!is.null(fit$pi_hat)) fit$pi_hat else 0
    freq_pred <- (1 - pi_hat) * lambda_pred
  } else if (dist == "nb") {
    # p_j regression, then E[N] = r_hat * (1 - p_j) / p_j
    p_pred <- as.vector(predict(fit$freq_glm, newdata=data.frame(age=ages), type="response"))
    r_hat <- fit$r_hat
    freq_pred <- r_hat * (1 - p_pred) / p_pred
  }
  # Severity prediction
  if (sevtype == "gamma") {
    # β_j regression, mean = α_hat * β_j
    beta_pred <- as.vector(predict(fit$sev_glm, newdata=data.frame(age=ages), type="response"))
    alpha_hat <- fit$alpha_hat
    sev_pred <- alpha_hat * beta_pred
  } else if (sevtype == "lognormal") {
    # μ_j regression, mean = exp(μ_j + 0.5σ^2)
    mu_pred <- as.vector(predict(fit$sev_glm, newdata=data.frame(age=ages)))
    sigma_hat <- fit$sigma_hat
    sev_pred <- exp(mu_pred + 0.5 * sigma_hat^2)
  }
  list(freq_pred = freq_pred, sev_pred = sev_pred)
}


evaluate_all_glm_models_implied <- function(fits, X, freq, sev_list) {
  model_names <- c("Poisson-Gamma", "ZIP-Gamma", "NB-Gamma", "Poisson-Lognormal", "ZIP-Lognormal", "NB-Lognormal")
  model_keys  <- c("poisson_gamma", "zip_gamma", "nb_gamma", "poisson_lognorm", "zip_lognorm", "nb_lognorm")
  dists       <- c("poisson", "zip", "nb", "poisson", "zip", "nb")
  sevs        <- c("gamma", "gamma", "gamma", "lognormal", "lognormal", "lognormal")
  
  obs_freq     <- freq
  obs_sev_mean <- sapply(sev_list, function(x) if(length(x) > 0) mean(x) else NA_real_)
  obs_agg      <- sapply(sev_list, sum)
  
  rmse <- function(est, true) {
    idx <- which(!is.na(est) & !is.na(true))
    if (length(idx) == 0) return(NA_real_)
    sqrt(mean((est[idx] - true[idx])^2))
  }
  stats <- matrix(NA, nrow=6, ncol=3)
  for (j in seq_along(model_keys)) {
    fit <- fits[[model_keys[j]]]
    pred <- get_policy_predictions_glm_implied(fit, X[,1], dists[j], sevs[j])
    stats[j,1] <- rmse(pred$freq_pred, obs_freq)
    idx_nonzero <- which(obs_freq > 0 & !is.na(obs_sev_mean))
    stats[j,2] <- rmse(pred$sev_pred[idx_nonzero], obs_sev_mean[idx_nonzero])
    pred_agg <- pred$freq_pred * pred$sev_pred
    stats[j,3] <- rmse(pred_agg, obs_agg)
  }
  colnames(stats) <- c("RMSE_Frequency", "RMSE_Severity", "RMSE_Aggregate")
  rownames(stats) <- model_names
  as.data.frame(stats)
}


glm_results <- fit_all_glm_models_implied(X_train, freq_train, sev_list_train)
rmse_table <- evaluate_all_glm_models_implied(glm_results, X_val, freq_val, sev_list_val)
print(round(rmse_table, 3))
