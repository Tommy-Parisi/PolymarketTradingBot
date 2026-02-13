use crate::model::valuation::CandidateTrade;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct AllocationConfig {
    pub max_trades_per_cycle: usize,
    pub max_fraction_per_trade: f64,
    pub max_total_fraction_per_cycle: f64,
    pub min_fraction_per_trade: f64,
    pub enforce_event_mutex: bool,
}

impl Default for AllocationConfig {
    fn default() -> Self {
        Self {
            max_trades_per_cycle: 5,
            max_fraction_per_trade: 0.06,
            max_total_fraction_per_cycle: 0.20,
            min_fraction_per_trade: 0.005,
            enforce_event_mutex: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AllocatedTrade {
    pub candidate: CandidateTrade,
    pub bankroll_fraction: f64,
    pub notional: f64,
}

pub struct PortfolioAllocator {
    cfg: AllocationConfig,
}

impl PortfolioAllocator {
    pub fn new(cfg: AllocationConfig) -> Self {
        Self { cfg }
    }

    pub fn allocate(&self, bankroll: f64, mut candidates: Vec<CandidateTrade>) -> Vec<AllocatedTrade> {
        if bankroll <= 0.0 || candidates.is_empty() {
            return Vec::new();
        }

        candidates.sort_by(|a, b| {
            let sa = a.edge_pct * a.confidence.max(0.0);
            let sb = b.edge_pct * b.confidence.max(0.0);
            sb.total_cmp(&sa)
        });

        let mut remaining_cycle_fraction = self.cfg.max_total_fraction_per_cycle;
        let mut out = Vec::new();
        let mut seen_event_keys: HashSet<String> = HashSet::new();
        for c in candidates.into_iter().take(self.cfg.max_trades_per_cycle) {
            if remaining_cycle_fraction <= 0.0 {
                break;
            }
            if self.cfg.enforce_event_mutex {
                let event_key = event_key_from_ticker(&c.ticker);
                if seen_event_keys.contains(&event_key) {
                    continue;
                }
                seen_event_keys.insert(event_key);
            }
            let kelly = approximate_kelly_fraction(c.fair_price, c.observed_price);
            let mut fraction = kelly
                .min(self.cfg.max_fraction_per_trade)
                .min(remaining_cycle_fraction);
            if fraction < self.cfg.min_fraction_per_trade {
                continue;
            }

            // Confidence scales down capital deployment for uncertain signals.
            fraction *= c.confidence.clamp(0.2, 1.0);
            if fraction < self.cfg.min_fraction_per_trade {
                continue;
            }

            let notional = bankroll * fraction;
            out.push(AllocatedTrade {
                candidate: c,
                bankroll_fraction: fraction,
                notional,
            });
            remaining_cycle_fraction -= fraction;
        }
        out
    }
}

fn event_key_from_ticker(ticker: &str) -> String {
    match ticker.rfind('-') {
        Some(i) if i > 0 => ticker[..i].to_string(),
        _ => ticker.to_string(),
    }
}

fn approximate_kelly_fraction(fair_price: f64, observed_price: f64) -> f64 {
    let p = fair_price.clamp(0.01, 0.99);
    let q = 1.0 - p;
    let b = (1.0 - observed_price).max(0.01) / observed_price.max(0.01);
    (((b * p) - q) / b).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::types::Side;
    use crate::model::valuation::CandidateTrade;

    fn c(edge: f64, conf: f64, fair: f64, obs: f64) -> CandidateTrade {
        CandidateTrade {
            ticker: "KX".to_string(),
            side: Side::Buy,
            outcome_id: "yes".to_string(),
            fair_price: fair,
            observed_price: obs,
            edge_pct: edge,
            confidence: conf,
            rationale: "x".to_string(),
        }
    }

    #[test]
    fn allocator_caps_total_fraction() {
        let alloc = PortfolioAllocator::new(AllocationConfig {
            max_trades_per_cycle: 10,
            max_fraction_per_trade: 0.06,
            max_total_fraction_per_cycle: 0.10,
            min_fraction_per_trade: 0.001,
            enforce_event_mutex: true,
        });
        let out = alloc.allocate(
            10_000.0,
            vec![c(0.2, 1.0, 0.70, 0.50), c(0.18, 1.0, 0.69, 0.50), c(0.16, 1.0, 0.68, 0.50)],
        );
        let total: f64 = out.iter().map(|x| x.bankroll_fraction).sum();
        assert!(total <= 0.10 + 1e-9);
    }

    #[test]
    fn allocator_ignores_tiny_positions() {
        let alloc = PortfolioAllocator::new(AllocationConfig {
            min_fraction_per_trade: 0.02,
            ..AllocationConfig::default()
        });
        let out = alloc.allocate(10_000.0, vec![c(0.08, 0.3, 0.52, 0.50)]);
        assert!(out.is_empty());
    }

    #[test]
    fn allocator_enforces_event_mutex_on_same_event_root() {
        let alloc = PortfolioAllocator::new(AllocationConfig {
            max_trades_per_cycle: 5,
            min_fraction_per_trade: 0.001,
            enforce_event_mutex: true,
            ..AllocationConfig::default()
        });
        let mut a = c(0.20, 1.0, 0.70, 0.50);
        a.ticker = "KXTABLE-AAA-BBB-AAA".to_string();
        let mut b = c(0.19, 1.0, 0.69, 0.50);
        b.ticker = "KXTABLE-AAA-BBB-BBB".to_string();
        let out = alloc.allocate(10_000.0, vec![a, b]);
        assert_eq!(out.len(), 1);
    }
}
