"""
Difference-in-Differences (DiD) Causal Inference Pipeline for AI Deployments

Author: Lucas Nogueira
Description: Simulates weekly ticket handling times, estimates the DiD interaction effect 
             via statsmodels OLS, verifies parallel trends, and plots queue trajectories.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def generate_did_data(n_weeks: int = 16, intervention_week: int = 9, seed: int = 42) -> pd.DataFrame:
    """Simulates weekly support ticket logs for Treatment and Control groups."""
    np.random.seed(seed)
    records = []
    
    # Baseline resolution times in minutes (Tier 1 starts higher due to queue backlog)
    base_treat = 22.0
    base_control = 17.0
    
    # The true unobserved causal effect of the AI bot (-7.0 minutes)
    true_ai_effect = -7.0
    
    for week in range(1, n_weeks + 1):
        # Global operational noise (e.g., platform outage pushing times up over time)
        global_trend = 0.35 * week 
        
        is_post = 1 if week >= intervention_week else 0
        
        # 200 tickets generated per queue each week
        for _ in range(200):
            # Treatment Group (Tier 1 - Receives GenAI Routing Bot in Week 9)
            err_treat = np.random.normal(0, 3.0)
            y_treat = base_treat + global_trend + (true_ai_effect * is_post) + err_treat
            records.append({
                'week': week,
                'group': 'Treatment (Tier 1)',
                'treat': 1,
                'post': is_post,
                'resolution_time': y_treat
            })
            
            # Control Group (Tier 2 - No AI Intervention throughout)
            err_control = np.random.normal(0, 3.0)
            y_control = base_control + global_trend + err_control
            records.append({
                'week': week,
                'group': 'Control (Tier 2)',
                'treat': 0,
                'post': is_post,
                'resolution_time': y_control
            })
            
    return pd.DataFrame(records)


def fit_did_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fits the Ordinary Least Squares (OLS) DiD model with interaction terms."""
    # Statsmodels formula notation automatically handles main effects and the interaction (treat:post)
    formula = "resolution_time ~ treat + post + treat:post"
    model = smf.ols(formula=formula, data=df).fit()
    return model


def plot_did_trajectories(df: pd.DataFrame, intervention_week: int, save_path: str = None) -> None:
    """Plots weekly resolution time trajectories with pre/post intervention markers."""
    weekly_avg = df.groupby(['week', 'group'])['resolution_time'].mean().reset_index()
    
    plt.figure(figsize=(11, 6), dpi=300)
    
    colors = {'Treatment (Tier 1)': '#d62728', 'Control (Tier 2)': '#1f77b4'}
    
    for grp in weekly_avg['group'].unique():
        subset = weekly_avg[weekly_avg['group'] == grp]
        plt.plot(subset['week'], subset['resolution_time'], marker='o', 
                 linewidth=2.2, label=grp, color=colors[grp])
        
    # Vertical marker for AI rollout
    plt.axvline(x=intervention_week - 0.5, color='#333333', linestyle='--', 
                linewidth=1.8, label='GenAI Deployment')
    
    # Highlight pre-treatment window for parallel trend visual inspection
    plt.axvspan(1, intervention_week - 0.5, color='#e0e0e0', alpha=0.35, label='Pre-Rollout Window')
    
    plt.title('Difference-in-Differences: Ticket Resolution Time Trajectory', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Week of Year', fontsize=11)
    plt.ylabel('Mean Resolution Time (Minutes)', fontsize=11)
    plt.xticks(range(1, df['week'].max() + 1))
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved successfully to {save_path}")
        
    plt.show()


if __name__ == "__main__":
    print("Generating synthetic support logs...")
    df_tickets = generate_did_data(n_weeks=16, intervention_week=9)
    
    print("Fitting OLS Difference-in-Differences Model...")
    model_fit = fit_did_model(df_tickets)
    
    print("\n" + "="*65)
    print("REGRESSION MODEL SUMMARY")
    print("="*65)
    print(model_fit.summary())
    
    plot_did_trajectories(df_tickets, intervention_week=9, save_path="assets/did_trajectories.png")
