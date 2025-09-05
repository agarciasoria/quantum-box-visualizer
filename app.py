import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Title and description
st.title("⚛️ Quantum Particle in a Box Visualizer")
st.write("Visualizing wavefunctions ψₙ(x) and probability densities |ψₙ(x)|²")

# Sidebar controls
st.sidebar.header("📊 Controls")
L = st.sidebar.slider("Box Length (L)", min_value=1, max_value=10, value=5, step=1)
n = st.sidebar.slider("Quantum Number (n)", min_value=1, max_value=10, value=4, step=1)
show_box = st.sidebar.checkbox("Show box walls", value=True)
show_equations = st.sidebar.checkbox("Show equations", value=True)

# Display equations
if show_equations:
    st.markdown("### 📐 Mathematical Description")
    col1, col2 = st.columns(2)
    
    with col1:
        st.latex(r"\psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)")
        st.caption("Wavefunction")
    
    with col2:
        st.latex(r"E_n = \frac{n^2 \pi^2 \hbar^2}{2mL^2}")
        st.caption("Energy eigenvalue")

# Physics calculations
x = np.linspace(0, L, 1000)
psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
prob = psi**2
E = (n**2 * np.pi**2) / (2 * L**2)  # Energy in units where ħ²/2m = 1

# Create the plot
fig = go.Figure()

# Add box walls if enabled
if show_box:
    # Left wall
    fig.add_shape(type="line",
        x0=0, y0=-1, x1=0, y1=1.5,
        line=dict(color="gray", width=4))
    
    # Right wall
    fig.add_shape(type="line",
        x0=L, y0=-1, x1=L, y1=1.5,
        line=dict(color="gray", width=4))
    
    # Box bottom
    fig.add_shape(type="line",
        x0=0, y0=-1, x1=L, y1=-1,
        line=dict(color="gray", width=2))
    
    # Add shaded regions outside box
    fig.add_shape(type="rect",
        x0=-1, y0=-1, x1=0, y1=1.5,
        fillcolor="rgba(128,128,128,0.2)",
        line=dict(color="rgba(0,0,0,0)"))
    
    fig.add_shape(type="rect",
        x0=L, y0=-1, x1=L+1, y1=1.5,
        fillcolor="rgba(128,128,128,0.2)",
        line=dict(color="rgba(0,0,0,0)"))

# Add wavefunction
fig.add_trace(go.Scatter(
    x=x, y=psi, 
    mode='lines', 
    name=f'ψ_{n}(x)',
    line=dict(color='blue', width=3),
    hovertemplate='x = %{x:.2f}<br>ψ = %{y:.3f}'
))

# Add probability density
fig.add_trace(go.Scatter(
    x=x, y=prob, 
    mode='lines', 
    name=f'|ψ_{n}(x)|²',
    line=dict(color='red', width=3),
    hovertemplate='x = %{x:.2f}<br>|ψ|² = %{y:.3f}'
))

# Add zero line
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)

# Style the plot
fig.update_layout(
    title=dict(
        text=f"Quantum State n={n}, Energy E={E:.3f}",
        font=dict(size=20)
    ),
    xaxis=dict(
        title="Position (x)",
        range=[-0.5, L+0.5],
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)'
    ),
    yaxis=dict(
        title="Amplitude",
        range=[-1.2, 1.5],
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)'
    ),
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        x=0.7,
        y=0.95,
        bgcolor='rgba(255,255,255,0.8)'
    ),
    plot_bgcolor='white'
)

# Display plot
st.plotly_chart(fig, use_container_width=True)

# Info boxes
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Energy Level", f"n = {n}")

with col2:
    st.metric("Energy", f"E_{n} = {E:.3f}", 
              help="In units where ħ²/2m = 1")

with col3:
    st.metric("Wavelength", f"λ = {2*L/n:.3f}")

# Additional information
with st.expander("📚 Learn More"):
    st.markdown("""
    ### Physical Interpretation
    
    - **ψₙ(x)** (blue): The quantum wavefunction - represents the quantum state
    - **|ψₙ(x)|²** (red): Probability density - likelihood of finding the particle at position x
    - **Gray walls**: Infinite potential barriers (particle cannot exist outside)
    - **n**: Quantum number - determines the energy level and number of nodes
    
    ### Key Observations
    
    1. Higher n → more nodes in the wavefunction
    2. The particle can never be found at the walls (ψ = 0 at x = 0 and x = L)
    3. Energy increases as n² (quadratically)
    4. The ground state (n=1) has no nodes inside the box
    """)

st.markdown("---")
st.caption("Created with ❤️ using Streamlit | Quantum Mechanics Visualization")