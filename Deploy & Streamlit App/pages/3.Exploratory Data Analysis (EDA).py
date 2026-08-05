import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title(" Exploratory Data Analysis (EDA)")

# ---------------------------
# LOAD FROM SESSION
# ---------------------------
if "df" not in st.session_state:
    st.warning(" Please load data first from the Data Preview section.")
    st.stop()

df = st.session_state.df.copy()

# ---------------------------
# BASIC CLEANING
# ---------------------------
df = df.dropna(how="all")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if len(numeric_cols) == 0:
    st.error(" No numeric columns found in dataset.")
    st.stop()

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header(" Controls")

x_axis = st.sidebar.selectbox("Select X-axis", numeric_cols)
y_axis = st.sidebar.selectbox("Select Y-axis", numeric_cols)

st.sidebar.subheader(" Filter")

filter_col = st.sidebar.selectbox("Filter Column", df.columns)

unique_vals = df[filter_col].dropna().unique()

selected_vals = st.sidebar.multiselect(
    "Select Values",
    unique_vals,
    default=unique_vals)

if len(selected_vals) == 0:
    st.warning("No filter selected → showing full dataset")
    filtered_df = df.copy()
else:
    filtered_df = df[df[filter_col].isin(selected_vals)]

# ---------------------------
# METRICS
# ---------------------------
st.subheader(" Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", filtered_df.shape[0])
col2.metric("Columns", filtered_df.shape[1])
col3.metric("Missing Values", int(filtered_df.isna().sum().sum()))

# ---------------------------
# DATA PREVIEW
# ---------------------------
st.subheader(" Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)

# ---------------------------
# SUMMARY STATISTICS
# ---------------------------
st.subheader(" Summary Statistics")
st.dataframe(filtered_df.describe(), use_container_width=True)

# ---------------------------
# SCATTER PLOT
# ---------------------------
st.subheader(" Scatter Plot")

fig1, ax1 = plt.subplots()
ax1.scatter(filtered_df[x_axis], filtered_df[y_axis])
ax1.set_xlabel(x_axis)
ax1.set_ylabel(y_axis)
st.pyplot(fig1)

# ---------------------------
# HISTOGRAM
# ---------------------------
st.subheader(" Histogram")

hist_col = st.selectbox("Select column for histogram", numeric_cols)

fig2, ax2 = plt.subplots()
ax2.hist(filtered_df[hist_col].dropna(), bins=20)
ax2.set_xlabel(hist_col)
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# ---------------------------
# CORRELATION HEATMAP (Matplotlib)
# ---------------------------
st.subheader(" Correlation Matrix")

corr = filtered_df[numeric_cols].dropna().corr()

fig3, ax3 = plt.subplots(figsize=(10,6))
cax = ax3.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax3.set_xticks(range(len(numeric_cols)))
ax3.set_yticks(range(len(numeric_cols)))
ax3.set_xticklabels(numeric_cols, rotation=90)
ax3.set_yticklabels(numeric_cols)

fig3.colorbar(cax)
st.pyplot(fig3)

# ---------------------------
# ADVANCED EDA (SEABORN + PLOTLY)
# ---------------------------
st.subheader(" Advanced EDA Insights")

# Optional: define model columns safely
MODEL_COLUMNS = numeric_cols[:min(6, len(numeric_cols))]  # limit to avoid overload
TARGET_COLUMN = numeric_cols[-1]  # assume last column as target (can customize)

# Seaborn Heatmap
st.markdown("** Detailed Correlation (Seaborn)**")
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.heatmap(filtered_df[MODEL_COLUMNS + [TARGET_COLUMN]].corr(), annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

# Feature Distributions
st.markdown("** Feature Distributions**")
for col in MODEL_COLUMNS:
    fig = px.histogram(filtered_df, x=col, marginal="box", nbins=30, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Scatter Matrix
st.markdown("** Feature Relationships (Scatter Matrix)**")

scatter_fig = px.scatter_matrix(
    filtered_df,
    dimensions=MODEL_COLUMNS,
    color=TARGET_COLUMN if TARGET_COLUMN in filtered_df.columns else None,
    height=800,
    title="Feature Relationships")

scatter_fig.update_traces(diagonal_visible=False)
st.plotly_chart(scatter_fig, use_container_width=True)

# ---------------------------
# DOWNLOAD DATA
# ---------------------------
st.subheader(" Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_data.csv",
    mime="text/csv")

