#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import minimize

print("Setup OK ")


# In[3]:


import numpy as np
import pandas as pd

np.random.seed(42)

n_weeks = 156  # 3 years weekly
dates = pd.date_range("2022-01-02", periods=n_weeks, freq="W")

t = np.arange(n_weeks)

# Seasonality + trend + holidays (controls)
trend = 1.2 * t
season = 80 * np.sin(2 * np.pi * t / 52) + 35 * np.cos(2 * np.pi * t / 26)

holiday = np.zeros(n_weeks)
holiday_weeks = [50, 51, 52, 103, 104, 105, 155]  # year-end peaks
holiday[holiday_weeks] = 1

def spend_series(base, vol, burst_prob=0.08, burst_mult=2.5):
    s = base + vol * np.abs(np.random.normal(size=n_weeks))
    bursts = (np.random.rand(n_weeks) < burst_prob).astype(float)
    s = s * (1 + bursts * (burst_mult - 1))
    return s

# Weekly channel spends
spend_search  = spend_series(200, 70, burst_prob=0.08, burst_mult=2.2)
spend_social  = spend_series(150, 55, burst_prob=0.10, burst_mult=2.0)
spend_display = spend_series(110, 45, burst_prob=0.09, burst_mult=2.3)
spend_email   = spend_series(35,  14, burst_prob=0.14, burst_mult=1.7)

# Base sales (without marketing)
base_sales = 1800 + trend + season + 220*holiday + np.random.normal(0, 40, n_weeks)

df = pd.DataFrame({
    "date": dates,
    "t": t,
    "holiday": holiday,
    "season": season,
    "trend": trend,
    "spend_search": spend_search,
    "spend_social": spend_social,
    "spend_display": spend_display,
    "spend_email": spend_email,
    "sales_base": base_sales
})

df.head()


# In[4]:


# ==========================================
# SECTION 2: CREATE REALISTIC MARKETING IMPACT
# ==========================================

def adstock(x, rate=0.5):
    """
    Geometric adstock:
    today's effect = today's spend + (rate * yesterday's effect)
    rate closer to 1 means longer carryover.
    """
    x = np.array(x, dtype=float)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + rate * out[i-1]
    return out

def saturation(x, alpha=0.0008):
    """
    Diminishing returns:
    increases quickly at first, then flattens as x grows.
    """
    x = np.array(x, dtype=float)
    return 1 - np.exp(-alpha * x)

# True underlying response (unknown to our model later)
search_eff  = 520 * saturation(adstock(df["spend_search"].values,  rate=0.60), alpha=0.0009)
social_eff  = 380 * saturation(adstock(df["spend_social"].values,  rate=0.50), alpha=0.0010)
display_eff = 260 * saturation(adstock(df["spend_display"].values, rate=0.40), alpha=0.0011)
email_eff   = 180 * saturation(adstock(df["spend_email"].values,   rate=0.30), alpha=0.0025)

noise = np.random.normal(0, 55, len(df))

df["sales"] = df["sales_base"] + search_eff + social_eff + display_eff + email_eff + noise

df[["date","sales","sales_base","spend_search","spend_social","spend_display","spend_email"]].head()


# In[5]:


# =========================
# SECTION 3: QUICK PLOT CHECK
# =========================

import matplotlib.pyplot as plt

plt.figure()
plt.plot(df["date"], df["sales"])
plt.title("Weekly Sales (with marketing effects)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


# In[6]:


# ==========================================
# SECTION 4: BUILD MMM FEATURES (MODEL INPUTS)
# ==========================================

# Reuse the same helper functions
def adstock(x, rate=0.5):
    x = np.array(x, dtype=float)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + rate * out[i-1]
    return out

def saturation(x, alpha=0.001):
    x = np.array(x, dtype=float)
    return 1 - np.exp(-alpha * x)

# These are "our MMM assumptions" (we choose reasonable values)
# In a full MMM project, you'd tune these or estimate them with Bayesian methods.
channel_params = {
    "search":  {"adstock": 0.60, "alpha": 0.0009},
    "social":  {"adstock": 0.50, "alpha": 0.0010},
    "display": {"adstock": 0.40, "alpha": 0.0011},
    "email":   {"adstock": 0.30, "alpha": 0.0025},
}

df_feat = df.copy()

df_feat["x_search"]  = saturation(adstock(df_feat["spend_search"].values,  channel_params["search"]["adstock"]),
                                  channel_params["search"]["alpha"])
df_feat["x_social"]  = saturation(adstock(df_feat["spend_social"].values,  channel_params["social"]["adstock"]),
                                  channel_params["social"]["alpha"])
df_feat["x_display"] = saturation(adstock(df_feat["spend_display"].values, channel_params["display"]["adstock"]),
                                  channel_params["display"]["alpha"])
df_feat["x_email"]   = saturation(adstock(df_feat["spend_email"].values,   channel_params["email"]["adstock"]),
                                  channel_params["email"]["alpha"])

# Model will use these + control variables
feature_cols = ["x_search", "x_social", "x_display", "x_email", "holiday", "trend", "season"]
df_feat[feature_cols + ["sales"]].head()


# In[7]:


# ==========================================
# SECTION 5: TRAIN/TEST SPLIT (TIME SERIES)
# ==========================================

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

split_idx = int(0.80 * len(df_feat))  # 80% train, 20% test

train = df_feat.iloc[:split_idx].copy()
test  = df_feat.iloc[split_idx:].copy()

X_train = train[feature_cols].values
y_train = train["sales"].values

X_test = test[feature_cols].values
y_test = test["sales"].values

print("Train weeks:", len(train), "Test weeks:", len(test))


# In[8]:


# ==========================================
# SECTION 6: FIT RIDGE MMM + EVALUATE
# ==========================================

model = Ridge(alpha=5.0)  # alpha controls regularization strength
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test  = model.predict(X_test)

print("Train R2:", round(r2_score(y_train, pred_train), 3), " | Train MAE:", round(mean_absolute_error(y_train, pred_train), 2))
print("Test  R2:", round(r2_score(y_test, pred_test), 3),  " | Test  MAE:", round(mean_absolute_error(y_test, pred_test), 2))

# Plot actual vs predicted (test period)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(test["date"], y_test, label="Actual")
plt.plot(test["date"], pred_test, label="Predicted")
plt.title("MMM-lite: Actual vs Predicted (Test)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()

# Show learned coefficients
coef_table = pd.DataFrame({
    "feature": feature_cols,
    "coef": model.coef_
}).sort_values("coef", ascending=False)

coef_table


# In[9]:


# ==========================================
# SECTION 7A: COMPUTE CHANNEL CONTRIBUTIONS
# ==========================================

coef_map = dict(zip(feature_cols, model.coef_))
intercept = model.intercept_

contrib = pd.DataFrame(index=df_feat.index)
contrib["date"] = df_feat["date"]

# Channel contributions
contrib["search"]  = coef_map["x_search"]  * df_feat["x_search"]
contrib["social"]  = coef_map["x_social"]  * df_feat["x_social"]
contrib["display"] = coef_map["x_display"] * df_feat["x_display"]
contrib["email"]   = coef_map["x_email"]   * df_feat["x_email"]

# Control contributions
contrib["holiday"] = coef_map["holiday"] * df_feat["holiday"]
contrib["trend"]   = coef_map["trend"]   * df_feat["trend"]
contrib["season"]  = coef_map["season"]  * df_feat["season"]

# Baseline (intercept)
contrib["baseline"] = intercept

# Total reconstructed sales
contrib["sales_reconstructed"] = contrib[
    ["search","social","display","email","holiday","trend","season","baseline"]
].sum(axis=1)

contrib.head()


# In[10]:


# ==========================================
# SECTION 7B: PLOT CHANNEL CONTRIBUTIONS
# ==========================================

plt.figure()
plt.stackplot(
    contrib["date"],
    contrib["search"],
    contrib["social"],
    contrib["display"],
    contrib["email"],
    labels=["Search","Social","Display","Email"]
)
plt.legend(loc="upper left")
plt.title("Weekly Sales Contribution by Channel")
plt.xlabel("Date")
plt.ylabel("Sales Contribution")
plt.tight_layout()
plt.show()


# In[11]:


# ==========================================
# SECTION 7C: TOTAL CONTRIBUTION SUMMARY
# ==========================================

total_contrib = contrib[["search","social","display","email"]].sum()

share_contrib = (total_contrib / total_contrib.sum()) * 100

summary = pd.DataFrame({
    "total_contribution": total_contrib,
    "share_percent": share_contrib
}).sort_values("total_contribution", ascending=False)

summary


# In[12]:


contrib.to_csv("outputs/weekly_contributions.csv", index=False)
summary.to_csv("outputs/channel_contribution_summary.csv")


# In[21]:


# =========================
# SECTION 8 FIXED BLOCK
# =========================

from scipy.optimize import minimize
import numpy as np

# Response function
def channel_response(spend, coef, adstock_rate, alpha):
    ad = adstock(np.array([spend]), adstock_rate)[0]
    sat = saturation(np.array([ad]), alpha)[0]
    return coef * sat

# Coefs from model
coefs = {
    "search":  coef_map["x_search"],
    "social":  coef_map["x_social"],
    "display": coef_map["x_display"],
    "email":   coef_map["x_email"],
}

# IMPORTANT: keys match function args now (adstock_rate, alpha)
params = {
    "search":  {"adstock_rate": channel_params["search"]["adstock"],  "alpha": channel_params["search"]["alpha"]},
    "social":  {"adstock_rate": channel_params["social"]["adstock"],  "alpha": channel_params["social"]["alpha"]},
    "display": {"adstock_rate": channel_params["display"]["adstock"], "alpha": channel_params["display"]["alpha"]},
    "email":   {"adstock_rate": channel_params["email"]["adstock"],   "alpha": channel_params["email"]["alpha"]},
}

def total_sales(spend_vec):
    spend_search, spend_social, spend_display, spend_email = spend_vec

    total = 0
    total += channel_response(spend_search,  coefs["search"],  **params["search"])
    total += channel_response(spend_social,  coefs["social"],  **params["social"])
    total += channel_response(spend_display, coefs["display"], **params["display"])
    total += channel_response(spend_email,   coefs["email"],   **params["email"])

    return -total  # minimize negative => maximize total

# Budget and constraints
avg_budget = (
    df["spend_search"].mean()
  + df["spend_social"].mean()
  + df["spend_display"].mean()
  + df["spend_email"].mean()
)

bounds = [
    (0.5 * df["spend_search"].mean(),  1.5 * df["spend_search"].mean()),
    (0.5 * df["spend_social"].mean(),  1.5 * df["spend_social"].mean()),
    (0.5 * df["spend_display"].mean(), 1.5 * df["spend_display"].mean()),
    (0.5 * df["spend_email"].mean(),   1.5 * df["spend_email"].mean()),
]

constraint = {"type": "eq", "fun": lambda x: np.sum(x) - avg_budget}

x0 = [
    df["spend_search"].mean(),
    df["spend_social"].mean(),
    df["spend_display"].mean(),
    df["spend_email"].mean(),
]

result = minimize(
    total_sales,
    x0=x0,
    bounds=bounds,
    constraints=[constraint],
    method="SLSQP"
)

result.success, result.message


# In[22]:


opt_spend = pd.Series(result.x, index=["Search","Social","Display","Email"])
current_spend = pd.Series({
    "Search":  df["spend_search"].mean(),
    "Social":  df["spend_social"].mean(),
    "Display": df["spend_display"].mean(),
    "Email":   df["spend_email"].mean(),
})

comparison = pd.DataFrame({
    "current_spend": current_spend,
    "optimized_spend": opt_spend,
    "change_%": 100 * (opt_spend - current_spend) / current_spend
})

comparison


# In[23]:


comparison.round(2).to_csv("outputs/final_budget_recommendation.csv")


# In[26]:


# ==========================================
# SECTION 9A: SIMULATE USER-LEVEL CAUSAL DATA
# ==========================================

import numpy as np
import pandas as pd

np.random.seed(42)

n_users = 20000

uplift_df = pd.DataFrame({
    "user_id": range(n_users),
    "prior_engagement": np.random.beta(2, 5, n_users),      # 0..1
    "price_sensitivity": np.random.beta(3, 3, n_users),     # 0..1
    "is_returning": np.random.binomial(1, 0.4, n_users)     # 0/1
})

# Randomized treatment assignment (simulating exposure)
uplift_df["treated"] = np.random.binomial(1, 0.5, n_users)

# Base conversion probability (without treatment)
base_conv = (
    0.02
    + 0.16 * uplift_df["prior_engagement"]
    + 0.05 * uplift_df["is_returning"]
    - 0.05 * uplift_df["price_sensitivity"]
)

# True heterogeneous treatment effect (uplift varies by user)
true_uplift = (
    0.10 * uplift_df["prior_engagement"]
    + 0.04 * uplift_df["is_returning"]
    - 0.06 * uplift_df["price_sensitivity"]
)

# Observed conversion probability
uplift_df["conversion_prob"] = np.clip(
    base_conv + uplift_df["treated"] * true_uplift,
    0, 1
)

# Observed outcome (converted = 1/0)
uplift_df["converted"] = np.random.binomial(1, uplift_df["conversion_prob"])

uplift_df.head()


# In[27]:


# ==========================================
# SECTION 9B: TRAIN UPLIFT MODELS (TWO-MODEL)
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

features = ["prior_engagement", "price_sensitivity", "is_returning"]

train_df, test_df = train_test_split(uplift_df, test_size=0.25, random_state=42, stratify=uplift_df["treated"])

treated_train = train_df[train_df["treated"] == 1]
control_train = train_df[train_df["treated"] == 0]

model_treated = RandomForestClassifier(max_depth=6, n_estimators=200, random_state=42)
model_control = RandomForestClassifier(max_depth=6, n_estimators=200, random_state=42)

model_treated.fit(treated_train[features], treated_train["converted"])
model_control.fit(control_train[features], control_train["converted"])

# Predict probabilities for test set (counterfactual style)
test_df = test_df.copy()
test_df["p_treated"] = model_treated.predict_proba(test_df[features])[:, 1]
test_df["p_control"] = model_control.predict_proba(test_df[features])[:, 1]
test_df["uplift_pred"] = test_df["p_treated"] - test_df["p_control"]

test_df[["p_treated", "p_control", "uplift_pred"]].head()


# In[28]:


# ==========================================
# SECTION 9C: UPLIFT VALIDATION BY BUCKET
# ==========================================

test_df["uplift_bucket"] = pd.qcut(test_df["uplift_pred"], 5, labels=False)

bucket_rows = []
for b in sorted(test_df["uplift_bucket"].unique()):
    d = test_df[test_df["uplift_bucket"] == b]
    conv_t = d[d["treated"] == 1]["converted"].mean()
    conv_c = d[d["treated"] == 0]["converted"].mean()
    bucket_rows.append({
        "bucket": int(b),
        "treated_conv_rate": conv_t,
        "control_conv_rate": conv_c,
        "observed_uplift": conv_t - conv_c,
        "n_users": len(d)
    })

uplift_bucket_summary = pd.DataFrame(bucket_rows).sort_values("bucket")
uplift_bucket_summary


# In[29]:


# ==========================================
# SECTION 9D: PLOT OBSERVED UPLIFT BY BUCKET
# ==========================================

import matplotlib.pyplot as plt

plt.figure()
plt.plot(uplift_bucket_summary["bucket"], uplift_bucket_summary["observed_uplift"], marker="o")
plt.title("Observed Incremental Lift by Predicted Uplift Bucket")
plt.xlabel("Uplift Bucket (low -> high)")
plt.ylabel("Observed Uplift (treated - control)")
plt.tight_layout()
plt.show()


# In[30]:


# ==========================================
# SECTION 9E: UPLIFT CURVE (TARGET TOP X%)
# ==========================================

d = test_df.sort_values("uplift_pred", ascending=False).reset_index(drop=True)

fractions = np.linspace(0.1, 1.0, 10)  # top 10%, 20%, ... 100%
curve = []

for f in fractions:
    k = int(f * len(d))
    top = d.iloc[:k]

    conv_t = top[top["treated"] == 1]["converted"].mean()
    conv_c = top[top["treated"] == 0]["converted"].mean()

    curve.append({
        "target_fraction": f,
        "observed_uplift": conv_t - conv_c,
        "n_users": k
    })

uplift_curve = pd.DataFrame(curve)

plt.figure()
plt.plot(uplift_curve["target_fraction"], uplift_curve["observed_uplift"], marker="o")
plt.title("Uplift Curve: Targeting Top Uplift Users")
plt.xlabel("Fraction of users targeted (top uplift)")
plt.ylabel("Observed Uplift (treated - control)")
plt.tight_layout()
plt.show()

uplift_curve


# In[31]:


# ==========================================
# SECTION 9F: EXPORT CAUSAL OUTPUTS
# ==========================================

uplift_bucket_summary.to_csv("outputs/uplift_bucket_summary.csv", index=False)
uplift_curve.to_csv("outputs/uplift_curve.csv", index=False)

print("Saved outputs/uplift_bucket_summary.csv and outputs/uplift_curve.csv ✅")


# In[ ]:




