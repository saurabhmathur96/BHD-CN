import numpy as np
from scipy.special import polygamma, digamma, loggamma
from scipy.optimize import fixed_point


class HierarchicalDirichletModel:
  def __init__(self, X_card, n_groups, s, s0):
    """
      X_card: tuple of ints
      n_groups: int
      s: float
      alpha0:
    """
    self.column_card = X_card
    self.X_card = np.prod(X_card)
    self.n_groups = n_groups
    self.s = s*self.X_card
    self.alpha0 = s0*np.ones(self.X_card)

    self.theta = np.zeros((self.X_card, self.n_groups))
    self.nu = np.ones((self.X_card, self.n_groups))
    self.tau = 1
    self.kappa = np.ones(self.X_card)/self.X_card

  def fit(self, X, groups):
    counts = np.zeros((self.X_card, self.n_groups)) + 1e-6
    for g in range(self.n_groups):
      group_mask = (groups == g)
      for i, row in enumerate(np.ndindex(*self.column_card)):
        counts[i, g] = np.sum(np.all(X == row, axis=1) & group_mask)

    self.counts = counts

    nu, tau, kappa = self.nu, self.tau, self.kappa

    def update_tau_kappa(x, nu):
      tau, kappa = x[0], x[1:]
      t_new = fixed_point(lambda t: self.tau_update(nu, t, kappa), x0=tau, method="iteration")
      k_new = fixed_point(lambda k: self.kappa_update(nu, tau, k), x0=kappa, method="iteration")
      return np.array([t_new, *k_new])

    def opt_tau_kappa(nu, tau, kappa):
      x0 = np.array([tau, *kappa])

      x = fixed_point(update_tau_kappa, x0=x0, args=(nu,), method="iteration")
      tau, kappa = x[0], x[1:]
      return tau, kappa
    
    def update(x, counts):
      shape = self.X_card, self.n_groups
      k = np.prod(shape)
      nu, tau, kappa = x[0:k].reshape(shape), x[k], x[k+1:]

      tau, kappa = opt_tau_kappa(nu, tau, kappa)
      nu = self.nu_update(nu, tau, kappa, counts)
      return np.array([*(nu.reshape(-1)), tau, *kappa])


    x0 = np.array([*(self.nu.reshape(-1)), self.tau, *self.kappa])

    x = fixed_point(update, x0=x0, method="iteration", args=(counts,))
    shape = self.X_card, self.n_groups
    k = np.prod(shape)
    nu, tau, kappa = x[0:k].reshape(shape), x[k], x[k+1:]
    self.nu, self.tau, self.kappa = nu, tau, kappa
    E_alpha = self.s*kappa[:, None]

    self.E_alpha = E_alpha

    self.theta = (counts + E_alpha)/(np.sum(counts, axis=0, keepdims=True)+self.s)
    self.E_alpha = self.E_alpha.reshape(self.column_card)
    self.theta = self.theta.reshape((*self.column_card, self.n_groups))

    return self

  def tau_update(self, nu, tau, kappa):
    g = 0
    first = (polygamma(1, tau*kappa)*kappa - polygamma(1, tau))
    second = (self.alpha0 - tau*kappa - self.n_groups*(self.s*kappa-1))
    g += np.sum(first*second)
    g += (self.s/(tau**2))*self.n_groups*(self.X_card - 1)

    h=0
    first = (polygamma(2, tau*kappa)*(kappa**2) - polygamma(2, tau))
    second = (self.alpha0 - tau*kappa - self.n_groups*(self.s*kappa - 1))
    h += np.sum(first*second)
    h += -np.sum(polygamma(1, tau*kappa)*(kappa**2))

    h += polygamma(1, tau) - (2*self.s/(tau**3))*self.n_groups*(self.X_card - 1)
    if np.isclose(g, 0):
      return tau
    delta = -g/(h*tau+g)
    return tau*np.exp(delta)

  def kappa_update(self, nu, tau, kappa):
    g = np.zeros_like(kappa)
    first = tau*polygamma(1, tau*kappa)
    second = (self.alpha0 - tau*kappa - self.n_groups*(self.s*kappa-1))
    g += first*second
    g += -self.s*self.n_groups*(digamma(tau*kappa) + digamma(self.s*kappa) -np.log(kappa) -1)
    g += -self.n_groups/kappa + np.sum(digamma(nu), axis=1)

    h = np.zeros_like(kappa)
    first = (tau*2)*polygamma(2, tau*kappa)
    second = (self.alpha0 - tau*kappa - self.n_groups*(self.s*kappa-1))
    h += first*second
    h += -tau*polygamma(1, tau*kappa)*(tau + 2*self.s*self.n_groups)
    h += -(self.s**2)*self.n_groups*polygamma(1, self.s*kappa)
    h += self.s*self.n_groups/kappa + self.n_groups/(kappa**2)

    numerator = np.sum(g/h)
    denominator = np.sum(1/h)

    delta = (numerator/denominator - g)/h

    return kappa + delta

  def nu_update(self, nu, tau, kappa, counts):
    return self.s*kappa[:, None] + counts

  def objective(self, nu, tau, kappa, counts):
    s0 = np.sum(self.alpha0)
    difference = digamma(nu) - digamma(np.sum(nu, axis = 0, keepdims = True))
    objective = 0

    objective  = np.sum(counts*difference)
    objective += np.sum((self.s*kappa - 1)[:, None]*difference)

    objective += np.sum(loggamma(nu))
    objective += -np.sum((nu-1)*difference)
    objective += -np.sum(loggamma(np.sum(nu, axis=0, keepdims=True)))

    objective += -self.n_groups*np.sum(loggamma(self.s*kappa))
    objective += self.n_groups*np.sum((self.s*kappa-1)*(np.log(kappa)-digamma(tau*kappa)+digamma(tau)))

    objective += -np.sum(loggamma(self.alpha0))
    objective += np.sum(loggamma(tau*kappa))
    objective += np.sum((self.alpha0 - tau*kappa)*(digamma(tau*kappa) - digamma(tau)))

    objective += self.n_groups*np.sum(loggamma(self.s))
    objective += -(self.s/tau)*self.n_groups*(self.X_card-1)
    objective += np.sum(loggamma(s0))
    objective += -np.sum(loggamma(tau))

    return objective

  @property
  def prior_correlation(self):
    s0 = np.sum(self.alpha0)
    return self.s/(s0 + self.s + 1)

