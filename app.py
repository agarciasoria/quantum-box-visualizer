import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Title
st.title("⚛️ Quantum Particle in a Box Visualizer")
st.write("Visualizing wavefunctions ψₙ(x) and probability densities |ψₙ(x)|²")

# Sidebar controls
st.sidebar.header("Controls")
L = st.sidebar.slider("Box Length (L)", min_value=1, max_value=10, value=5, step=1)
n = st.sidebar.slider("Quantum Number (n)", min_value=1, max_value=10, value=1, step=1)

# Physics calculations
x = np.linspace(0, L, 1000)  # x-axis points
psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)  # Wavefunction
prob = psi**2  # Probability density
E = (n**2 * np.pi**2) / (2 * L**2)  # Energy (ħ=1, m=1)

# Create the plot
fig = go.Figure()

# Add wavefunction
fig.add_trace(go.Scatter(
    x=x, y=psi, 
    mode='lines', 
    name=f'ψ_{n}(x)',
    line=dict(color='blue', width=3)
))

# Add probability density
fig.add_trace(go.Scatter(
    x=x, y=prob, 
    mode='lines', 
    name=f'|ψ_{n}(x)|²',
    line=dict(color='red', width=3)
))

# Style the plot
fig.update_layout(
    title=f"Quantum State n={n}, Energy E={E:.2f}",
    xaxis_title="Position (x)",
    yaxis_title="Amplitude",
    hovermode='x',
    showlegend=True
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Info box
st.info(f"Energy Eigenvalue: E_{n} = {E:.2f} (in units where ħ²/2m = 1)")