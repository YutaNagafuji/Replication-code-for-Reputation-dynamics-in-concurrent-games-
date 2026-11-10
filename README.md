# Reputation Dynamics in Concurrent Games

This repository provides the official Python replication code for the evolutionary stability analysis and figure reproduction in the paper:

> **Reputation dynamics in concurrent games**  
> Yuta Nagafuji, Eizo Akiyama, and Ryoji Sawa (2026)  
> *Working Paper*

---

## 1. Overview

The codebase conducts an exhaustive computational search over all $2^8 = 256$ second-order social norms and $(2^4)^2 = 256$ dual-game action strategies under various reputation fusion mechanisms (AND, OR, and Probabilistic $f_Q$), reproducing the analytical and numerical findings in the paper.

### Repository Structure

```text
.
├── concurrent_games_ESS_simulation.py  # Core simulation engine (ESS enumeration & filtering)
├── plot_figure1.py                     # Standalone replication script for Figure 1
├── plot_max_payoff_by_Q.py             # Plotting script for Figure 3 & Figure 4 (Q vs Max Payoff)
├── requirements.txt                    # Python library dependencies
├── LICENSE                             # MIT License
└── README.md                           # Documentation and replication guide
```

---

## 2. Requirements & Installation

The code is tested on **Python 3.8+** with standard scientific computing libraries:

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `matplotlib>=3.4.0`

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---
### 3. Mathematical Formulations & Implementation

The implementation strictly reflects the model described in the manuscript:

1. **Reputation Transitions (Eqs. 11–12)**: Accounts for independent action execution errors $\mu_e$ and reputation assessment errors $\mu_a$ for each game interaction.
2. **Reputation Fusion (Eqs. 13–15)**:
   - **AND rule ($f_A$, Eq. 13)**:
     $$D_{h \hat{h}_1 \hat{h}_2}^s = D_{h \hat{h}_1}^{s, 1} \cdot D_{h \hat{h}_2}^{s, 2}$$
   - **OR rule ($f_O$, Eq. 14)**:
     $$D_{h \hat{h}_1 \hat{h}_2}^s = 1 - (1 - D_{h \hat{h}_1}^{s, 1})(1 - D_{h \hat{h}_2}^{s, 2})$$
   - **Probabilistic rule ($f_Q$, Eq. 15)**:
     $$D_{h \hat{h}_1 \hat{h}_2}^s = (1 - Q) D_{h \hat{h}_1}^{s, 1} + Q D_{h \hat{h}_2}^{s, 2}$$
3. **Stationary Reputation $H^*$ (Eq. 16)**:
   - Evaluates the cubic characteristic polynomial for resident steady-state good reputation $H^*(s) \in [0, 1]$.
   - *Note on multi-stability*: If multiple valid real roots exist in $[0, 1]$, the branch maximizing resident expected payoff $\pi(s|s)$ is tracked.
   - Mutant reputation $H^*(s')$ is solved analytically as:
     $$H^*(s') = \frac{F}{1 - E + F}$$
4. **Realized Cooperation (Eq. 20)**: Incorporates execution noise via the affine transformation:
   $$\tilde{Y}_i(\cdot, \cdot) = \mu_e + (1 - 2\mu_e) Y_i(\cdot, \cdot)$$
5. **Evolutionary Stability (Definition 2, Eq. 23)**: Computes resident payoff $\pi(s|s)$ (Eq. 21) and mutant payoff $\pi(s'|s)$ (Eq. 22). Evaluates whether:
   $$\pi(s|s) > \max_{s' \ne s} \pi(s'|s) + \varepsilon$$
   against all 255 alternative mutant strategies.
6. **Classification & Filtering (Definition 1)**:
   - **Cooperative**: $d_{11}^C = 1$ and $s_{11}^1 = s_{11}^2 = C$.
   - **Semi-cooperative**: $d_{11}^C = 1$ and ($s_{11}^1 = C$ or $s_{11}^2 = C$).

---

## 4. How to Run the Simulations

### 4.1 Exhaustive ESS Search (`concurrent_games_ESS_simulation.py`)

Run the script directly from the terminal:

```bash
python concurrent_games_ESS_simulation.py
```

#### Parameter Customization
Parameters can be configured directly inside the `main()` function:

```python
# 1. Fusion rule: "AND", "OR", or "Q"
fusion_mode = "AND"

# 2. Game payoffs: (b^1, c^1) and (b^2, c^2) (Table 1)
b1, c1 = 2.1, 1.0
b2, c2 = 2.1, 1.0

# 3. Error rates: mu_e = mu_a = mu (Section 2.4)
mu_e, mu_a = 0.01, 0.01

# 4. ESS strictness tolerance
eps = 1e-8

# 5. Grid of Q values (when fusion_mode="Q")
Q_list = [round(i * 0.01, 2) for i in range(101)]

# 6. CPU multiprocessing workers
nproc = 4
```

#### Output Files
Execution produces two CSV files:
1. `ESS_8state_[MODE]_...csv`: All configurations $(d, s)$ satisfying the ESS condition.
2. `ESS_8state_[MODE]_..._cooperative.csv`: Subset filtered strictly by Definition 1 ($d_{11}^C = 1, s_{11}^1 = s_{11}^2 = C$) and sorted by descending expected payoff.

---

## 5. Reproducing the Figures

### 5.1 Reproducing Figure 1 (Stability Regions in $(b^1, b^2)$ Parameter Space)

`plot_figure1.py` evaluates stability against all 255 mutant strategies across a $101 \times 101$ grid of $(b^1, b^2) \in [0, 10] \times [0, 10]$ ($c^1=c^2=1, \mu=0.01$) for:
- All 8 Leading Eight (L8) configurations under AND fusion.
- All 6 Sustainable Six (S6) configurations under OR fusion.

Run:
```bash
python plot_figure1.py
```
This generates `Figure1_reproduced.png` displaying the three regions:
- **Red**: Sustainable Six (OR) is evolutionarily stable and payoff-maximizing.
- **Blue**: Leading Eight (AND) is evolutionarily stable and payoff-maximizing.
- **White**: Mutual defection (ALLD) only.

---

### 5.2 Reproducing Figure 3 & Figure 4 (Maximum Payoff vs. Weight $Q$)

Figures 3 and 4 plot the maximum resident expected payoff against the Game 2 weight $Q \in [0, 1]$ under probabilistic fusion $f_Q$.

#### Step 1: Run Simulation for the Target Payoff Structure
Set `fusion_mode = "Q"` in `concurrent_games_ESS_simulation.py` with the corresponding payoffs, for example:
- **Figure 3a**: $b^1 = b^2 = 1.1$, $c^1 = c^2 = 1.0$
- **Figure 3c**: $b^1 = b^2 = 2.0$, $c^1 = c^2 = 1.0$
- **Figure 4a**: $b^1 = 1.1, c^1 = 1.0$, $b^2 = 2.1, c^2 = 2.0$

Run the simulation:
```bash
python concurrent_games_ESS_simulation.py
```

#### Step 2: Generate the Plot
Pass the generated CSV file to `plot_max_payoff_by_Q.py`:

```bash
python plot_max_payoff_by_Q.py \
    ESS_8state_Q_b1_1p1_c1_1_b2_1p1_c2_1_mue_0p01_mua_0p01_eps_1e-08_Q0to1n101.csv \
    -o Figure3a.png
```

*Note: For $Q$ values where no cooperative ESS exists, the payoff defaults to $0.0$ (ALLD), matching the curves in Figures 3 and 4.*

---

## 6. CSV Data Dictionary

| Column Header | Manuscript Symbol | Description |
| :--- | :--- | :--- |
| `fusion_rule` | $f \in \{f_A, f_O, f_Q\}$ | Reputation fusion rule (`AND`, `OR`, or `Q`). |
| `Q` | $Q \in [0, 1]$ | Reputational weight of Game 2 under probabilistic fusion $f_Q$. |
| `norm_d` | $d \in \{0, 1\}^8$ | Binary social norm vector $(d_{11}^C, d_{11}^D, d_{10}^C, d_{10}^D, d_{01}^C, d_{01}^D, d_{00}^C, d_{00}^D)$. |
| `strategy_s1` | $s^1 \in \{C, D\}^4$ | Behavioral strategy in Game 1 $(s_{11}^1, s_{10}^1, s_{01}^1, s_{00}^1)$. |
| `strategy_s2` | $s^2 \in \{C, D\}^4$ | Behavioral strategy in Game 2 $(s_{11}^2, s_{10}^2, s_{01}^2, s_{00}^2)$. |
| `is_cooperative` | Definition 1 | `True` if $d_{11}^C = 1$ and $s_{11}^1 = s_{11}^2 = C$. |
| `is_semi_cooperative` | Definition 1 | `True` if $d_{11}^C = 1$ and ($s_{11}^1 = C$ or $s_{11}^2 = C$). |
| `payoff_resident` | $\pi(s\|s)$ (Eq. 21) | Total stationary expected payoff of resident players. |
| `payoff_resident_game1` | $\pi^1(s\|s)$ | Expected resident payoff obtained in Game 1. |
| `payoff_resident_game2` | $\pi^2(s\|s)$ | Expected resident payoff obtained in Game 2. |
| `payoff_resident_relative` | $\frac{\pi(s\|s)}{\sum_i (b^i - c^i)}$ | Normalized payoff relative to error-free cooperation. |
| `H_star_resident` | $H^*(s)$ (Eq. 16) | Stationary fraction of good unified reputations among residents. |
| `H_roots_all` | - | All valid real roots of the characteristic polynomial in $[0, 1]$. |
| `num_H_roots` | - | Number of valid real roots found. |
| `payoff_mutant_max` | $\max_{s' \ne s} \pi(s'\|s)$ | Highest expected payoff achieved among all 255 mutant strategies. |
| `stability_margin` | $\pi(s\|s) - \max_{s'} \pi(s'\|s)$ | Margin of evolutionary stability ($> \varepsilon$ for strict ESS). |
| `worst_mutant_s1` | $\hat{s}^{1*}$ | Game 1 strategy of the most threatening mutant. |
| `worst_mutant_s2` | $\hat{s}^{2*}$ | Game 2 strategy of the most threatening mutant. |
| `H_star_worst_mutant` | $H^*(\hat{s}^*)$ | Stationary good reputation fraction of the worst mutant. |

---

## 7. Citation

If you use this code in academic work, please cite:

```bibtex
@article{nagafuji2026reputation,
  title={Reputation dynamics in concurrent games},
  author={Nagafuji, Yuta and Akiyama, Eizo and Sawa, Ryoji},
  journal={Working Paper},
  year={2026}
}
```

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
