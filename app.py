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
with st.expander("📚 Learn More - Detailed Physics"):
    st.markdown("""
    ### 📐 Complete Derivation from Schrödinger Equation
    
    #### 1. Time-Independent Schrödinger Equation
    For a particle of mass m in a potential V(x), the time-independent Schrödinger equation is:
    """)
    
    st.latex(r"-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi")
    
    st.markdown("""
    #### 2. The Infinite Square Well Potential
    - Inside the box (0 < x < L): V(x) = 0
    - Outside the box: V(x) = ∞
    
    Since V = ∞ outside, ψ(x) = 0 for x ≤ 0 and x ≥ L (particle cannot exist there).
    
    #### 3. Solving Inside the Box
    For 0 < x < L, where V = 0, the equation becomes:
    """)
    
    st.latex(r"-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} = E\psi")
    
    st.markdown("Rearranging:")
    
    st.latex(r"\frac{d^2\psi}{dx^2} = -\frac{2mE}{\hbar^2}\psi = -k^2\psi")
    
    st.markdown("where we define:")
    
    st.latex(r"k = \sqrt{\frac{2mE}{\hbar^2}}")
    
    st.markdown("""
    #### 4. General Solution
    The general solution to this differential equation is:
    """)
    
    st.latex(r"\psi(x) = A\sin(kx) + B\cos(kx)")
    
    st.markdown("""
    #### 5. Applying Boundary Conditions
    
    **Boundary condition 1:** ψ(0) = 0
    """)
    
    st.latex(r"\psi(0) = A\sin(0) + B\cos(0) = B = 0")
    
    st.markdown("Therefore B = 0, and:")
    
    st.latex(r"\psi(x) = A\sin(kx)")
    
    st.markdown("**Boundary condition 2:** ψ(L) = 0")
    
    st.latex(r"\psi(L) = A\sin(kL) = 0")
    
    st.markdown("""
    Since A ≠ 0 (otherwise ψ = 0 everywhere), we need sin(kL) = 0.
    This happens when:
    """)
    
    st.latex(r"kL = n\pi, \quad n = 1, 2, 3, ...")
    
    st.markdown("""
    #### 6. Quantization Emerges!
    From kL = nπ, we get:
    """)
    
    st.latex(r"k = \frac{n\pi}{L}")
    
    st.markdown("Substituting back into the definition of k:")
    
    st.latex(r"\sqrt{\frac{2mE}{\hbar^2}} = \frac{n\pi}{L}")
    
    st.markdown("Solving for E:")
    
    st.latex(r"E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}")
    
    st.markdown("""
    **This is quantization!** Energy can only take discrete values determined by the quantum number n.
    
    #### 7. Normalizing the Wavefunction
    The wavefunction must be normalized:
    """)
    
    st.latex(r"\int_0^L |\psi(x)|^2 dx = 1")
    
    st.latex(r"\int_0^L A^2\sin^2\left(\frac{n\pi x}{L}\right) dx = 1")
    
    st.markdown("Using the integral identity ∫sin²(ax)dx = x/2 - sin(2ax)/(4a), we get:")
    
    st.latex(r"A^2 \cdot \frac{L}{2} = 1")
    
    st.latex(r"A = \sqrt{\frac{2}{L}}")
    
    st.markdown("""
    #### 8. Final Wavefunction
    The normalized energy eigenfunctions are:
    """)
    
    st.latex(r"\psi_n(x) = \sqrt{\frac{2}{L}}\sin\left(\frac{n\pi x}{L}\right)")
    
    st.markdown("""
    ---
    ### 🌊 About the Wavelength
    
    The de Broglie wavelength λ is related to momentum p by:
    """)
    
    st.latex(r"\lambda = \frac{h}{p} = \frac{2\pi\hbar}{p}")
    
    st.markdown("From the energy-momentum relation for a free particle:")
    
    st.latex(r"E = \frac{p^2}{2m}")
    
    st.markdown("We can express momentum as:")
    
    st.latex(r"p = \sqrt{2mE} = \sqrt{2m \cdot \frac{n^2\pi^2\hbar^2}{2mL^2}} = \frac{n\pi\hbar}{L}")
    
    st.markdown("Therefore, the de Broglie wavelength is:")
    
    st.latex(r"\lambda = \frac{2\pi\hbar}{p} = \frac{2\pi\hbar}{\frac{n\pi\hbar}{L}} = \frac{2L}{n}")
    
    st.markdown(f"""
    **For the current state (n = {n}):** λ = {2*L/n:.3f}
    
    Notice that L = nλ/2, meaning the box length is exactly n half-wavelengths!
    """)
    
    st.markdown("""
    ---
    ### 🎯 Key Physical Insights
    
    1. **Quantization Origin**: Boundary conditions (ψ = 0 at walls) restrict k to discrete values
    2. **Zero-Point Energy**: The lowest energy is n=1, not zero (E₁ = π²ℏ²/2mL²)
    3. **Node Pattern**: State n has (n-1) nodes inside the box
    4. **Orthogonality**: Different eigenstates are orthogonal: ∫ψₙψₘdx = 0 if n ≠ m
    5. **Uncertainty Principle**: As L decreases, E increases (confinement increases momentum uncertainty)
    """)

st.markdown("---")
st.caption("Created with ❤️ using Streamlit | Quantum Mechanics Visualization")