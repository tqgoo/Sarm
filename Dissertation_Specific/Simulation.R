# --- Libraries ---
library(keras)
library(tensorflow)
library(reticulate)
library(MASS)
library(knitr)
library(dplyr)
library(tibble)
np <- import("numpy")

# --- 0. NN-Sarmanov simulation and fit --------------------------
set.seed(369)
M <- 10000         # number of policies
alpha0 <- 2.87        # gamma shape (simulation)
omega0 <- 0.3      # true dependence

# -- Simulate data --
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
