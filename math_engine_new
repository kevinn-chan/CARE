import numpy as np

#environmental context
RISK_MATRIX = {
    "road_type": {
        "Complex Intersection": 1.2,
        "High-Speed Highway": 0.8,
        "Straight City Road": 0.4
    },
    "traffic_flow": {
        "Rush Hour (Gridlock)": 2.5,
        "Night (Low Visibility)": 2.0,
        "Normal Flow": 1.0
    },
    "surface_conditions": {
        "Heavy Rain/Wet Asphalt": 2.5,
        "Fog/Low Visibility": 1.8,
        "Clear/Dry": 1.0
    }
}

def _clip01(x, eps=1e-6):
    return np.clip(x, eps, 1.0 - eps)

#posterior_odds = prior_odds * (P(V|T) / P(V|¬T))^evidence_power
#if P(V|T) ~= P(V|¬T) (uncertain/out-of-domain), LR ~= 1, posterior ~= prior
#if P(V|T) >> P(V|¬T) (crash evidence), posterior rises sharply
def calculate_bayesian_posterior(prior, p_vgt, p_vgnt=None, *, evidence_power=1.5, eps=1e-6):
    prior = np.asarray(prior, dtype=float)
    p_vgt = np.asarray(p_vgt, dtype=float)

    prior = _clip01(prior, eps)
    p_vgt = _clip01(p_vgt, eps)

    #approximate P(V|¬T) as the complement if p_vgt is not provided
    #safe_prob_sum ~= 1 - threat_prob_sum
    if p_vgnt is None:
        p_vgnt = 1.0 - p_vgt
    p_vgnt = _clip01(np.asarray(p_vgnt, dtype=float), eps)

    lr = (p_vgt / p_vgnt) ** float(evidence_power)

    prior_odds = prior / (1.0 - prior)
    post_odds = prior_odds * lr
    post = post_odds / (1.0 + post_odds)

    return float(post) if post.size == 1 else post


def monte_carlo_sims(
    flow,
    weather,
    road="Straight City Road",
    sims=10000,
    *,
    #set what "normal traffic" looks like
    null_mean=0.20,
    null_strength=30.0,
    #how visual evidence changes the posterior
    evidence_power=1.5,
    #VaR percentile on posterior
    var_q=0.95,
    #prevent insane contexts from making the prior unrealistic
    max_prior=0.25,
    seed=None):

    #prior_pt: contextual prior P(T)
    #var_threshold: VaR threshold computed on the posterior under nominal conditions
    base_rate = RISK_MATRIX["road_type"].get(road, 0.4)
    flow_mult = RISK_MATRIX["traffic_flow"].get(flow, 1.0)
    noise_factor = RISK_MATRIX["surface_conditions"].get(weather, 1.0)

    total_arrival_rate = base_rate * flow_mult

    #prior distribution
    a = max(1e-3, total_arrival_rate * noise_factor)
    b = max(1e-3, (20.0 / noise_factor))

    rng = np.random.default_rng(seed)

    #simulate prior uncertainty
    prior_samples = rng.beta(a, b, size=sims)
    prior_samples = np.clip(prior_samples, 1e-6, max_prior)

    #simulate what the vision model outputs during normal traffic
    null_a = max(1e-3, null_mean * null_strength)
    null_b = max(1e-3, (1.0 - null_mean) * null_strength)
    p_vgt_null = rng.beta(null_a, null_b, size=sims)

    posterior_null = calculate_bayesian_posterior(
        prior_samples,
        p_vgt_null,
        evidence_power=evidence_power
    )

    prior_pt = float(np.mean(prior_samples))
    var_threshold = float(np.quantile(posterior_null, var_q))
    return prior_pt, var_threshold

#require N consecutive exceedances before declaring crash
class PersistenceGate:
    def __init__(self, consecutive=2):
        self.consecutive = int(consecutive)
        self._streak = 0

    def reset(self):
        self._streak = 0

    def update(self, value, threshold):
        if value > threshold:
            self._streak += 1
        else:
            self._streak = 0
        return self._streak >= self.consecutive
